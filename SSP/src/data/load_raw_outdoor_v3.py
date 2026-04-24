"""
load_raw_outdoor_v2.py
======================
포트홀 감지 XGBoost용 전처리 파이프라인 (개선판)
 
주요 변경사항:
  1. 라벨링: delta 고정 → 물리적 거리 기반 (속도 × 시간)
  2. 윈도우: stride=1 → stride=WINDOW_SIZE//2 (겹침 50%로 감소)
  3. 샘플링: 단순 다운샘플 → class_weight 전달 (데이터 손실 없음)
  4. 피처: Z축 편향 제거, 속도 보정, 충격 복원 패턴, 교차 축 피처 추가
  5. 평가: trip5 단일 → Leave-One-Trip-Out CV
"""
 
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
 
# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
WINDOW_SIZE = 20          # 윈도우 샘플 수
STRIDE      = 5          # 윈도우 이동 간격 (50% 겹침)
LABEL_RADIUS_M = 5.0     # 포트홀 라벨링 반경 (미터)
MIN_SPEED_MS   = 0.5      # 정지 상태 제외 기준 (m/s)
CORR = 0.7
# ─────────────────────────────────────────────────────────────────────────────
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
 
def find_trip_files(base_path: str):
    """trip*_sensors.csv / trip*_potholes.csv 자동 탐색 (파일 수 무관)"""
    sensor_files  = sorted(glob.glob(os.path.join(base_path, "trip*_sensors.csv")))
    pothole_files = sorted(glob.glob(os.path.join(base_path, "trip*_potholes.csv")))
    assert len(sensor_files) == len(pothole_files), \
        f"sensors({len(sensor_files)})와 potholes({len(pothole_files)}) 파일 수가 다릅니다."
    return sensor_files, pothole_files
 
 
def load_trip(sensor_path: str, pothole_path: str, trip_id: int):
    sensors  = pd.read_csv(sensor_path)
    potholes = pd.read_csv(pothole_path)
 
    sensors["trip_id"]  = trip_id
    potholes["trip_id"] = trip_id
 
    sensors  = sensors.sort_values("timestamp").reset_index(drop=True)
    potholes = potholes.sort_values("timestamp").reset_index(drop=True)
    return sensors, potholes
 
 
def load_all(base_path: str):
    sensor_files, pothole_files = find_trip_files(base_path)
    all_sensors, all_potholes = [], []
    for i, (sf, pf) in enumerate(zip(sensor_files, pothole_files), start=1):
        s, p = load_trip(sf, pf, trip_id=i)
        all_sensors.append(s)
        all_potholes.append(p)
        print(f"  trip{i}: sensors={len(s)}, potholes={len(p)}")
    return pd.concat(all_sensors, ignore_index=True), \
           pd.concat(all_potholes, ignore_index=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. 라벨링 — 물리적 거리 기반
# ══════════════════════════════════════════════════════════════════════════════
 
def haversine_m(lat1, lon1, lat2, lon2):
    """두 GPS 좌표 사이 거리 (미터)"""
    R = 6_371_000   # 지구 반지름(M)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)     # GPS 좌표(degree) -> radian 변환
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2     # 두 점 사이의 각 거리 비율 계산
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))     # arctan2 : 두 점 사이 중심각 | R * 중심각 = 실제거리

 
 
def assign_pothole_timestamps(sensors: pd.DataFrame,
                               potholes: pd.DataFrame) -> pd.DataFrame:
    """
    각 포트홀 타임스탬프를 센서 row에 GPS 거리 기반으로 매핑.
    sensors에 'is_near_pothole' 컬럼 추가.
 
    sensors에 latitude/longitude가 없으면 timestamp 기반 fallback 사용.
    """
    sensors = sensors.copy()
    sensors["is_near_pothole"] = False
 
    has_gps = "latitude" in sensors.columns and "longitude" in sensors.columns
 
    for _, ph_row in potholes.iterrows():
        ph_ts = ph_row["timestamp"]
 
        if has_gps and "latitude" in ph_row and "longitude" in ph_row:
            ph_lat, ph_lon = ph_row["latitude"], ph_row["longitude"]
            dist = haversine_m(
                sensors["latitude"].values,
                sensors["longitude"].values,
                ph_lat, ph_lon
            )
            mask = dist <= LABEL_RADIUS_M
        else:
            # GPS 없음: ±1.5초 fallback
            mask = (sensors["timestamp"] >= ph_ts - 1.5) & \
                   (sensors["timestamp"] <= ph_ts + 1.5)
 
        sensors.loc[mask, "is_near_pothole"] = True
 
    return sensors
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. 슬라이딩 윈도우 (stride 조정, 정지 상태 제외)
# ══════════════════════════════════════════════════════════════════════════════
 
def make_windows(sensors: pd.DataFrame):
    """
    trip별로 stride=STRIDE 슬라이딩 윈도우 생성.
    - 평균 속도 < MIN_SPEED_MS 인 윈도우 제외 (정지 상태)
    - 라벨: 윈도우 내 is_near_pothole True 비율 > THRESHOLD_RATE
    반환: windows, labels, window_trip_ids
    """
    THRESHOLD_RATE = 0.25
 
    all_windows, all_labels, all_trip_ids = [], [], []
 
    for tid in sorted(sensors["trip_id"].unique()):
        trip = sensors[sensors["trip_id"] == tid].reset_index(drop=True)
 
        for start in range(0, len(trip) - WINDOW_SIZE + 1, STRIDE):
            win = trip.iloc[start: start + WINDOW_SIZE]
 
            if win["speed"].mean() < MIN_SPEED_MS:
                continue
 
            label = int(win["is_near_pothole"].mean() >= THRESHOLD_RATE)
            all_windows.append(win)
            all_labels.append(label)
            all_trip_ids.append(tid)
 
    pos = sum(all_labels)
    print(f"  총 윈도우: {len(all_labels)}  |  포트홀(1): {pos}  |  정상(0): {len(all_labels)-pos}")
    return all_windows, all_labels, all_trip_ids
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. 피처 추출
# ══════════════════════════════════════════════════════════════════════════════
 
def _safe_skew(x):
    return float(skew(x)) if len(x) > 2 else 0.0
 
def _safe_kurt(x):
    return float(kurtosis(x)) if len(x) > 2 else 0.0
 
def _zero_crossings(x):
    return int(np.sum(np.diff(np.sign(x - x.mean())) != 0))
 
def _dominant_freq_energy(x, fs=10.0):
    """가장 강한 주파수 성분의 에너지 비율"""
    fft_mag = np.abs(np.fft.rfft(x - x.mean()))
    total   = fft_mag.sum() + 1e-9
    return float(fft_mag.max() / total)
 
def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)

def featurize_window(win: pd.DataFrame) -> dict:
    f = {}
    speed = win["speed"].values

    ax = win["accelerometerX"].values
    ay = win["accelerometerY"].values
    az = win["accelerometerZ"].values
    gx = win["gyroX"].values
    gy = win["gyroY"].values
    gz = win["gyroZ"].values

    mean_speed = np.clip(speed.mean(), MIN_SPEED_MS, None)
    az_norm = az / mean_speed

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    f["acc_mag_mean"] = acc_mag.mean()
    f["acc_mag_max"] = acc_mag.max()
    f["acc_mag_std"] = acc_mag.std()
    f["acc_mag_skew"] = _safe_skew(acc_mag)
    f["acc_mag_kurt"] = _safe_kurt(acc_mag)

    for name, arr in [("ax", ax), ("ay", ay), ("az", az),
                      ("gx", gx), ("gy", gy), ("gz", gz),
                      ("az_norm", az_norm)]:
        f[f"{name}_mean"] = arr.mean()
        f[f"{name}_std"] = arr.std()
        f[f"{name}_max"] = arr.max()
        f[f"{name}_min"] = arr.min()
        f[f"{name}_range"] = arr.max() - arr.min()
        f[f"{name}_skew"] = _safe_skew(arr)
        f[f"{name}_kurt"] = _safe_kurt(arr)
        f[f"{name}_zc"] = _zero_crossings(arr)

    peaks, props = find_peaks(np.abs(az), height=az.std(), distance=2)
    f["az_peak_count"] = len(peaks)
    f["az_peak_height_max"] = props["peak_heights"].max() if len(peaks) > 0 else 0.0

    if len(peaks) > 0:
        p = peaks[0]
        after = az[p:] if p < len(az) else az
        f["az_post_peak_min"] = after.min()
        f["az_post_peak_range"] = after.max() - after.min()
    else:
        f["az_post_peak_min"] = 0.0
        f["az_post_peak_range"] = 0.0

    f["az_energy"] = float(np.sum(az**2))
    f["az_norm_energy"] = float(np.sum(az_norm**2))
    f["az_fft_dom_ratio"] = _dominant_freq_energy(az)

    diff_az = np.diff(az)
    abs_diff_az = np.abs(diff_az)

    f["az_diff_std"] = diff_az.std() if len(diff_az) > 0 else 0.0
    f["az_diff_max"] = diff_az.max() if len(diff_az) > 0 else 0.0
    f["az_diff_min"] = diff_az.min() if len(diff_az) > 0 else 0.0
    f["az_abs_diff_mean"] = abs_diff_az.mean() if len(abs_diff_az) > 0 else 0.0
    f["az_abs_diff_max"] = abs_diff_az.max() if len(abs_diff_az) > 0 else 0.0
    f["az_abs_diff_std"] = abs_diff_az.std() if len(abs_diff_az) > 0 else 0.0

    dt = 0.2
    jerk_az = diff_az / dt if len(diff_az) > 0 else np.array([0.0])
    f["az_jerk_mean"] = float(np.mean(np.abs(jerk_az)))
    f["az_jerk_max"] = float(np.max(np.abs(jerk_az)))
    f["az_jerk_std"] = float(np.std(jerk_az))

    az_rms = _rms(az)
    f["az_rms"] = az_rms
    f["az_crest_factor"] = float(np.max(np.abs(az)) / az_rms)

    az_ma = pd.Series(az).rolling(window=3, min_periods=1, center=True).mean().values
    az_dev = az - az_ma
    f["az_ma_dev_mean"] = float(np.mean(np.abs(az_dev)))
    f["az_ma_dev_max"] = float(np.max(np.abs(az_dev)))
    f["az_ma_dev_std"] = float(np.std(az_dev))

    f["az_vs_axay_ratio"] = float(
        np.abs(az).mean() / (np.abs(ax).mean() + np.abs(ay).mean() + 1e-9)
    )
    f["acc_gyro_corr_xz"] = float(np.corrcoef(ax, gz)[0, 1]) if len(ax) > 2 else 0.0
    f["acc_gyro_corr_yz"] = float(np.corrcoef(ay, gz)[0, 1]) if len(ay) > 2 else 0.0

    f["speed_mean"] = speed.mean()
    f["speed_std"] = speed.std()
    f["speed_range"] = speed.max() - speed.min()

    f["az_peak_over_speed"] = float(np.max(np.abs(az)) / (mean_speed + 1e-9))
    f["az_rms_over_speed2"] = float(az_rms / ((mean_speed ** 2) + 1e-9))
    f["speed_x_az_std"] = float(mean_speed * np.std(az))
    f["speed_x_az_range"] = float(mean_speed * (np.max(az) - np.min(az)))
    f["speed_x_az_energy"] = float(mean_speed * np.sum(az**2))

    if mean_speed < 5:
        f["speed_bucket_low"] = 1
        f["speed_bucket_mid"] = 0
        f["speed_bucket_high"] = 0
    elif mean_speed < 15:
        f["speed_bucket_low"] = 0
        f["speed_bucket_mid"] = 1
        f["speed_bucket_high"] = 0
    else:
        f["speed_bucket_low"] = 0
        f["speed_bucket_mid"] = 0
        f["speed_bucket_high"] = 1

    return f
 
 
def featurize_all(windows: list) -> pd.DataFrame:
    rows = [featurize_window(w) for w in windows]
    return pd.DataFrame(rows)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. Leave-One-Trip-Out 교차 검증용 데이터셋 생성
# ══════════════════════════════════════════════════════════════════════════════
 
def make_loto_splits(feature_df: pd.DataFrame, X: pd.DataFrame,
                     labels: list, window_trip_ids: list):
    trip_ids = np.array(window_trip_ids)
    splits = []
    for tid in np.unique(trip_ids):
        test_mask  = trip_ids == tid
        train_mask = ~test_mask
        splits.append({
            "test_trip": tid,
            "X_train": X[train_mask].reset_index(drop=True),
            "y_train": np.array(labels)[train_mask],
            "X_test":  X[test_mask].reset_index(drop=True),
            "y_test":  np.array(labels)[test_mask],
        })
    for sp in splits:
        tid = sp["test_trip"]
        pos = sp["y_test"].sum()
        print(f"  test_trip={tid}: train={len(sp['y_train'])}, test={len(sp['y_test'])} (포트홀={pos})")
    return splits
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════
 
def build_dataset(base_path: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
 
    # 6-1. 로드
    print("▶ 1. 데이터 로드")
    sensors, potholes = load_all(base_path)
 
    # 6-2. 라벨링 (GPS 거리 기반)
    print("▶ 2. 포트홀 라벨링 (GPS 거리 기반)")
    labeled_trips = []
    for tid in sorted(sensors["trip_id"].unique()):
        s_trip = sensors[sensors["trip_id"] == tid].copy()
        p_trip = potholes[potholes["trip_id"] == tid].copy()
        labeled_trips.append(assign_pothole_timestamps(s_trip, p_trip))
    sensors = pd.concat(labeled_trips, ignore_index=True)
    print(f"  포트홀 근방 센서 rows: {sensors['is_near_pothole'].sum()}")
 
    # 6-3. 윈도우 생성
    print("▶ 3. 슬라이딩 윈도우 생성")
    windows, labels, window_trip_ids = make_windows(sensors)
 
    # 6-4. 피처 추출
    print("▶ 4. 피처 추출")
    feature_df = featurize_all(windows)
    feature_df["label"] = labels
    feature_df["trip_id"] = window_trip_ids
    feature_df = feature_df.dropna().reset_index(drop=True)
    labels = feature_df["label"].tolist()
    window_trip_ids = feature_df["trip_id"].tolist()
    X = feature_df.drop(columns=["label", "trip_id"])
 
    print(f"  최종 피처 수: {X.shape[1]}")
    print(f"  클래스 분포: {pd.Series(labels).value_counts().to_dict()}")
    
    # 컬럼간 상관관계 분석
    import seaborn as sns
    import matplotlib.pyplot as plt
    X_num = X.select_dtypes(include=[np.number]).copy()

    # 상관계수 행렬
    corr_matrix = X_num.corr()
    corr_abs = corr_matrix.abs()

    # 히트맵
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.3)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # upper triangle
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    # 높은 상관 쌍 추출
    high_corr_pairs = (
        upper.stack()
                .reset_index()
                .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
                .sort_values(by="corr", ascending=False)
    )

    high_corr_pairs = high_corr_pairs[high_corr_pairs["corr"] >= CORR]

    print(f"\n[상관계수 {CORR} 이상 피처 쌍]")
    print(high_corr_pairs)

    # 제거 후보
    to_drop = [column for column in upper.columns if any(upper[column] > CORR)]

    print(f"\n[제거 후보 피처 ({len(to_drop)}개)]")
    print(to_drop)

    X = X_num.drop(columns=to_drop)

    print("\n원본 shape :", X_num.shape)
    print("축소 shape :", X.shape)

    # x_test = test_data.drop("label", axis=1).drop(columns=to_drop)
    # y_test = test_data["label"]

    # 6-5. LOTO splits 생성
    print("▶ 5. Leave-One-Trip-Out splits 생성")
    splits = make_loto_splits(feature_df, X, labels, window_trip_ids)
 
    # 6-6. 전체 저장 (마지막 trip을 test로 고정한 단일 split도 함께 저장)
    # (a) LOTO용: trip_id 포함 전체 데이터 → train_xgb_v2.py에서 바로 사용
    full_df = X.copy()
    full_df["label"]   = labels
    full_df["trip_id"] = window_trip_ids   # make_windows에서 함께 반환
    full_df.to_csv(os.path.join(save_path, f"full_v3_s25.csv"), index=False)
 
    # (b) 기존 호환: 마지막 trip을 test로 고정한 단일 split
    last_split = splits[-1]
    last_split["X_train"].assign(label=last_split["y_train"]).to_csv(
        os.path.join(save_path, f"train_v3_s25.csv"), index=False
    )
    last_split["X_test"].assign(label=last_split["y_test"]).to_csv(
        os.path.join(save_path, f"test_v3_s25.csv"), index=False
    )
    print(f"  저장 완료 → {save_path}")
 
    return splits, feature_df

# ══════════════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
 
    BASE_PATH = os.path.join(project_root, "data", "raw",       "pothole")
    SAVE_PATH = os.path.join(project_root, "data", "processed", "pothole")
 
    splits, feature_df = build_dataset(BASE_PATH, SAVE_PATH)