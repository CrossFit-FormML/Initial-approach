import os
import pandas as pd

root_dir = r"../../VL/프레스/숄더프레스/팔움직임오류"
dataframes = []
session_counter = 0

# 제외할 열 목록
exclude_columns = [
    'Nose_x', 'Nose_y',
    'LEye_x', 'LEye_y',
    'REye_x', 'REye_y',
    'LEar_x', 'LEar_y',
    'REar_x', 'REar_y',
    'Head_x', 'Head_y',
    'Neck_x', 'Neck_y',
    'LSmallToe_x', 'LSmallToe_y',
    'RSmallToe_x', 'RSmallToe_y', 'image_filename'
]

# 하위 디렉토리 순회
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.startswith("Motion") and filename.endswith(".csv"):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)

                # 해당 열이 있을 경우만 제거
                df = df.drop(columns=[col for col in exclude_columns if col in df.columns])

                df['crossfit_label'] = 2  # 엉덩이하방자세 오류
                df['session_id'] = session_counter
                session_counter += 1

                dataframes.append(df)

            except Exception as e:
                print(f"❌ {file_path} 불러오기 실패: {e}")

# 병합 및 저장
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    if 'image_filename' in combined_df.columns:
        combined_df = combined_df.drop(columns=['image_filename'])
    combined_df.to_csv("merged_arm_error.csv", index=False)
    print(f"✅ 완료: 총 {len(combined_df)}개 프레임, {session_counter}개 세션 병합됨.")
else:
    print("❗ CSV 파일을 찾지 못했습니다.")
