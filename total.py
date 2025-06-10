import pandas as pd

# 파일 경로와 데이터프레임 이름 매핑
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
    
    # session_id offset 적용
    df['session_id'] += session_offset
    session_offset += df['session_id'].nunique()
    
    dataframes.append(df)

# 병합
merged_all = pd.concat(dataframes, ignore_index=True)

# 저장
merged_all.to_csv("combined_crossfit_data_with_session_filter.csv", index=False)
print(f"✅ 최종 병합 완료: 총 {len(merged_all)}개 프레임, 세션 수: {merged_all['session_id'].nunique()}")
