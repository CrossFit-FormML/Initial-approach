import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("result", exist_ok=True)  # 저장 폴더 생성


# ========================
# 0. 상대좌표 및 velocity 변환 함수 정의
# ========================
def to_relative_coords(df, ref_joint='Hip'):
    ref_x = df[f'{ref_joint}_x']
    ref_y = df[f'{ref_joint}_y']
    for col in df.columns:
        if '_x' in col:
            df[col] = df[col] - ref_x
        elif '_y' in col:
            df[col] = df[col] - ref_y
    return df

# ========================
# 1. 시퀀스 데이터 생성 (velocity 포함)
# ========================
def make_sliding_window_sequences(df, window_size=90, stride=30):
    sequences = []
    labels = []
    for sid in df['session_id'].unique():
        session = df[df['session_id'] == sid].reset_index(drop=True)
        features = session.drop(columns=['crossfit_label', 'session_id']).values
        targets = session['crossfit_label'].values

        for i in range(0, len(session) - window_size + 1, stride):
            seq = features[i:i+window_size]                         # (90, 34)
            vel = np.diff(seq, axis=0)                              # (89, 34)
            vel = np.vstack([np.zeros((1, seq.shape[1]) ), vel])   # (90, 34)
            full_seq = np.concatenate([seq, vel], axis=1)           # (90, 68)
            label = targets[i + window_size - 1]
            sequences.append(full_seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# ========================
# 2. 데이터 로드 및 전처리 (전체 클래스 균형 oversampling 포함)
# ========================
from imblearn.over_sampling import RandomOverSampler

filtered_df = pd.read_csv('combined_crossfit_data_with_session_filter.csv')
filtered_df = to_relative_coords(filtered_df.copy())
X, y = make_sliding_window_sequences(filtered_df, window_size=90, stride=30)

# 전체 클래스 균형을 맞추기 위한 oversampling
X_flat = X.reshape(X.shape[0], -1)
y = np.array(y)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y)

# 복원
X_balanced = X_resampled.reshape(-1, X.shape[1], X.shape[2])

# 셔플
shuffle_idx = np.random.permutation(len(y_resampled))
X_balanced = X_balanced[shuffle_idx]
y_balanced = y_resampled[shuffle_idx]

X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
y_tensor = torch.tensor(y_balanced, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.3, stratify=y_tensor, random_state=42
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)


# ========================
# 3. Hybrid 모델 정의 (Conv1D + Transformer)
# ========================
class HybridClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)                     # Conv1d expects (B, F, T)
        x = self.conv1(x)                         # (B, d_model, T)
        x = self.relu(x).transpose(1, 2)          # (B, T, d_model)
        x = self.norm(x)
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.classifier(x[:, 0])

# ========================
# 4. 학습 루프 + Scheduler + Label Smoothing
# ========================

# ========================
# 4. 학습 루프 + Scheduler + Label Smoothing + Early Stopping
# ========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 해당 부분 수정필요!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model = HybridClassifier(input_dim=X.shape[2], seq_len=X.shape[1], num_classes=4).to(device)

class_sample_count = np.bincount(y)
weights = 1. / torch.tensor(class_sample_count, dtype=torch.float32)
class_weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Early stopping 설정
num_epochs = 200
best_loss = float('inf')
patience = 10
patience_counter = 0

train_loss_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 조기 종료 및 최고 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "press_transformer_best.pth")
        print("✅ 모델 저장됨 (best)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

# ========================
# 5. 평가
# ========================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1).cpu()
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy())

print("\n[Classification Report]")
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hybrid Transformer Confusion Matrix")
plt.tight_layout()
plt.savefig("result/confusion_matrix_train.png")  # 저장
plt.show()

# Loss 시각화
plt.figure()
plt.plot(train_loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("result/train_loss_curve.png")  # 저장
plt.show()

