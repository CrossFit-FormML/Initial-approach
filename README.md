# Crossfit 프로젝트

## 1. 프로젝트 개요
크로스핏 운동 중 4가지 핵심 동작인 Deadlift, Press, Squat, Clean에서 발생하는 정상 자세와 오류자세를 분류하는 딥러닝 기반 시스템을 개발하는 것을 목표로 합니다.

## 2. 실험 코드 제출

### 주요 파일 구조
```
├── README.md                                    # 프로젝트 설명서
├── Sample.ipynb                                 # 메인 실험 노트북
├── result/                                      # 결과 폴더
├── combine_오버헤드.py                           # 오버헤드 동작 데이터 결합 스크립트
├── combine_정상.py                              # 정상 동작 데이터 결합 스크립트  
├── combine_팔꿈치팔어깨활성화.py                   # 팔꿈치/팔/어깨 활성화 데이터 결합 스크립트
├── combine_팔움직임.py                           # 팔 움직임 데이터 결합 스크립트
├── combined_crossfit_data_with_session_filter.... # 세션 필터링된 통합 크로스핏 데이터
├── crossfit-7_frame.csv                         # 크로스핏 7프레임 데이터
├── crossfit-8_frame.csv                         # 크로스핏 8프레임 데이터  
├── crossfit-9_frame.csv                         # 크로스핏 9프레임 데이터
├── merged_arm_error.csv                         # 팔 오류 통합 데이터
├── merged_elbow_shoulder_error.csv              # 팔꿈치/어깨 오류 통합 데이터
├── merged_good.csv                              # 정상 동작 통합 데이터
├── merged_overhead_error.csv                    # 오버헤드 오류 통합 데이터
├── press_transformer_best.pth                   # Press 운동 최적 Transformer 모델
├── test.py                                      # 테스트 스크립트
└── total.py                                     # 전체 데이터 통합 스크립트
```

### 데이터 라벨링 시스템
- **라벨 0**: 정상 동작 (Good)
- **라벨 1**: 팔꿈치/어깨 오류 (Elbow/Shoulder Error)  
- **라벨 2**: 오버헤드 오류 (Overhead Error)
- **라벨 3**: 팔 움직임 오류 (Arm Movement Error)

### 코드 실행 방법

1. **환경 설정**
   - 구글 코랩 A100 GPU 사용 (코랩에서 CSV 파일만 런타임에 넣어주시면 그대로 실행하셔도 됩니다!)
   - Python 3.11.13
   - 필요한 라이브러리: tensorflow, torch, pandas, numpy, scikit-learn, matplotlib 등

2. **실행 순서**
   ```bash
   # Jupyter Notebook 실행
   jupyter notebook Sample.ipynb
   ```

3. **노트북 실행 가이드**
   - 셀 단위로 순차적으로 실행
   - 각 운동별 데이터 로드 및 전처리
   - 모델 학습 및 평가
   - 결과 시각화 및 분석

### 데이터셋 정보

#### 데이터 출처
본 프로젝트에서 사용된 CrossFit 운동 데이터셋은 다음 소스를 기반으로 합니다:

**원본 데이터셋:**
- **데이터셋 명**: 크로스핏 동작 데이터셋
- **출처 URL**: [AI Hub 크로스핏 동작 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%ED%81%AC%EB%A1%9C%EC%8A%A4%ED%95%8F%20%EB%8F%99%EC%9E%91%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71422)

#### 데이터 구조
- **프레임 단위 데이터**: 7, 8, 9 프레임으로 구분된 시계열 데이터
- **관절 좌표**: 34개 관절점의 x, y 좌표 정보
  - LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
  - LHip, RHip, Hip, LKnee, RKnee, LAnkle, RAnkle
  - LHeel, RHeel, LBigToe, RBigToe
- **라벨링**: crossfit_label (0-3)
- **세션 ID**: session_id로 각 운동 세션 구분

#### 데이터 전처리
- **윈도우 슬라이딩**: 90프레임 윈도우, 30프레임 스트라이드
- **정규화**: 좌표 데이터 정규화 처리
- **세션 필터링**: 중복 세션 제거 및 데이터 품질 향상

### 모델 정보

#### Press 운동 모델 (press_transformer_best.pth)
- **모델 타입**: Hybrid Classifier (CNN + Transformer + LSTM)
- **입력 차원**: 34개 관절 좌표 (x, y)
- **시퀀스 길이**: 90 프레임
- **출력 클래스**: 4개 (정상, 팔꿈치/어깨 오류, 오버헤드 오류, 팔 움직임 오류)

### 실험 결과

실험 결과는 다음과 같은 내용을 포함합니다:
- 각 모델의 성능 지표 (정확도, 손실, F1-score 등)
- 교차 검증 결과
- 하이퍼파라미터 튜닝 기록
- 학습 곡선 데이터
- 혼동 행렬 (Confusion Matrix)

### 데이터 결합 스크립트

#### combine_정상.py
정상 동작 데이터를 통합하여 merged_good.csv 생성

#### combine_오버헤드.py  
오버헤드 오류 데이터를 통합하여 merged_overhead_error.csv 생성

#### combine_팔꿈치팔어깨활성화.py
팔꿈치/어깨 관련 오류 데이터를 통합하여 merged_elbow_shoulder_error.csv 생성

#### combine_팔움직임.py
팔 움직임 오류 데이터를 통합하여 merged_arm_error.csv 생성

#### total.py
모든 오류 유형 데이터를 최종 통합하여 combined_crossfit_data_with_session_filter.csv 생성

```python
# total.py 사용 예시
files = {
    "good": "merged_good.csv",
    "elbow": "merged_elbow_shoulder_error.csv", 
    "overhead": "merged_overhead_error.csv",
    "arm": "merged_arm_error.csv"
}

# 순차적으로 데이터프레임 로드 및 session_id 조정
dataframes = []
session_offset = 0

for name, path in files.items():
    df = pd.read_csv(path)
    df['session_id'] += session_offset
    session_offset += df['session_id'].nunique()
    dataframes.append(df)

# 병합
merged_all = pd.concat(dataframes, ignore_index=True)
merged_all.to_csv("combined_crossfit_data_with_session_filter.csv", index=False)
```

### 주요 특징

1. **실시간 자세 분석**: MediaPipe 기반 실시간 포즈 추출
2. **다중 오류 분류**: 4가지 오류 유형 동시 분류
3. **시계열 처리**: LSTM과 Transformer를 활용한 시계열 데이터 학습
4. **데이터 증강**: 슬라이딩 윈도우를 통한 데이터 증강
5. **세션 기반 분할**: 개인별 세션 단위로 데이터 분할하여 일반화 성능 향상

### 실행 요구사항

1. **데이터 전처리**: 각 운동별 CSV 파일들이 필요
2. **실행 환경**: CUDA 지원 GPU 환경에서 실행 시 학습 속도 향상  
3. **메모리 요구사항**: 대용량 데이터셋 처리를 위해 충분한 RAM 필요
4. **모델 파일**: press_transformer_best.pth 모델 파일 필요

### 향후 개선 방향

- 실시간 피드백 시스템 구축
- 더 많은 크로스핏 동작 추가
- 웹 연동
- 개인별 맞춤형 교정 시스템 개발
