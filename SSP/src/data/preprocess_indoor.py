import os
import math
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from typing import Tuple # 타입 힌팅을 위한 모듈 추가

# --------------------------------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# --------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

BASE_PATH = os.path.join(project_root, "data", "raw", "indoor", "career-con-2019")
X_TRAIN_PATH = os.path.join(BASE_PATH, "X_train.csv")
Y_TRAIN_PATH = os.path.join(BASE_PATH, "y_train.csv")
X_TEST_PATH = os.path.join(BASE_PATH, "X_test.csv")

OUTPUT_PATH = os.path.join(project_root, "data", "processed", "indoor")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    """
    재현성 확보를 위한 난수 시드 고정 함수 (project_rules.md 3항 준수)
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    쿼터니언(Quaternion) 데이터를 오일러 각도(Euler Angles)로 변환하는 함수.
    로봇의 자세(Roll, Pitch, Yaw)를 파악하기 위함.
    """
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 센서 데이터를 활용하여 새로운 물리적/통계적 피처를 생성하는 함수.
    """
    # 1. 쿼터니언 -> 오일러 각도 변환
    df['roll'], df['pitch'], df['yaw'] = zip(*df.apply(lambda x: quaternion_to_euler(x.orientation_X, x.orientation_Y, x.orientation_Z, x.orientation_W), axis=1))
    
    # 2. 벡터 크기(Magnitude) 계산
    # 가속도 및 각속도 통합 크기
    df['accel_mag'] = np.sqrt(df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)
    df['gyro_mag'] = np.sqrt(df['angular_velocity_X']**2 + df['angular_velocity_Y']**2 + df['angular_velocity_Z']**2)
    
    # 3. 변화량(Diff) 계산 (동적 특징 추출)
    df['accel_diff'] = df.groupby('series_id')['accel_mag'].diff().fillna(0)
    
    return df

def calc_range(x: pd.Series) -> float:
    """최대값과 최소값의 차이를 계산하는 명시적 함수 (컬럼 네이밍 꼬임 방지)"""
    return x.max() - x.min()

def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    128개의 측정 지점(series_id별)을 하나의 통계적 요약 행으로 변환하는 함수.
    """
    # 통계량을 산출할 컬럼 리스트
    target_cols = [
        'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
        'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z',
        'roll', 'pitch', 'yaw', 'accel_mag', 'gyro_mag', 'accel_diff'
    ]
    
    # 그룹화하여 다양한 통계량 계산 (lambda 대신 명시적 함수 calc_range 사용)
    agg_df = df.groupby('series_id')[target_cols].agg([
        'mean', 'std', 'max', 'min', 'median', 
        calc_range, skew, kurtosis
    ])
    
    # 컬럼 이름 정리 (예: accel_mag_mean)
    agg_df.columns = [f"{col}_{stat}" if not callable(stat) else f"{col}_{stat.__name__}" 
                      for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()
    
    return agg_df

def main() -> None:
    set_seed(42) # 파이프라인 재현성을 위한 시드 고정
    print("실내 데이터 전처리 시작...")
    
    # 데이터 불러오기
    print("1. 데이터 로딩 중...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    
    # 피처 엔지니어링 수행
    print("2. 피처 생성 중 (오일러 각도, 벡터 크기 등)...")
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    
    # 시퀀스별 통계 데이터 요약
    print("3. 시퀀스 단위 통계량 요약 중...")
    train_features = aggregate_features(X_train)
    test_features = aggregate_features(X_test)
    
    # 라벨 인코딩 (문자열 재질 -> 숫자)
    print("4. 라벨 인코딩 중...")
    le = LabelEncoder()
    y_train['surface_encoded'] = le.fit_transform(y_train['surface'])
    
    # 최종 데이터 병합
    train_final = train_features.merge(y_train, on='series_id')
    
    # 결과 저장
    print("5. 전처리 결과 저장 중...")
    train_final.to_csv(os.path.join(OUTPUT_PATH, "indoor_train_features.csv"), index=False)
    test_features.to_csv(os.path.join(OUTPUT_PATH, "indoor_test_features.csv"), index=False)
    
    # 인코딩 매핑 정보 출력
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"전처리 완료! 라벨 매핑: {mapping}")
    print(f"저장 경로: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()