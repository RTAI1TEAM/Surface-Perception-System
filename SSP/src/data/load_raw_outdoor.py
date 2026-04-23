import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import koreanize_matplotlib

WINDOW_SIZE = 20
THRESHOLD_RATE = 0.9

pd.set_option('display.max_columns', None)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

BASE_PATH = os.path.join(project_root, "data", "raw", "pothole")
TRIP1_POTHOLES_PATH = os.path.join(BASE_PATH, "trip1_potholes.csv")
TRIP2_POTHOLES_PATH = os.path.join(BASE_PATH, "trip2_potholes.csv")
TRIP3_POTHOLES_PATH = os.path.join(BASE_PATH, "trip3_potholes.csv")
TRIP4_POTHOLES_PATH = os.path.join(BASE_PATH, "trip4_potholes.csv")
TRIP5_POTHOLES_PATH = os.path.join(BASE_PATH, "trip5_potholes.csv")
TRIP1_SENSORS_PATH = os.path.join(BASE_PATH, "trip1_sensors.csv")
TRIP2_SENSORS_PATH = os.path.join(BASE_PATH, "trip2_sensors.csv")
TRIP3_SENSORS_PATH = os.path.join(BASE_PATH, "trip3_sensors.csv")
TRIP4_SENSORS_PATH = os.path.join(BASE_PATH, "trip4_sensors.csv")
TRIP5_SENSORS_PATH = os.path.join(BASE_PATH, "trip5_sensors.csv")

def load_data():
    raw_potholes_data1 = pd.read_csv(TRIP1_POTHOLES_PATH)
    raw_potholes_data2 = pd.read_csv(TRIP2_POTHOLES_PATH)
    raw_potholes_data3 = pd.read_csv(TRIP3_POTHOLES_PATH)
    raw_potholes_data4 = pd.read_csv(TRIP4_POTHOLES_PATH)

    raw_sensors_data1 = pd.read_csv(TRIP1_SENSORS_PATH)
    raw_sensors_data2 = pd.read_csv(TRIP2_SENSORS_PATH)
    raw_sensors_data3 = pd.read_csv(TRIP3_SENSORS_PATH)
    raw_sensors_data4 = pd.read_csv(TRIP4_SENSORS_PATH)

    # Test용 데이터
    raw_potholes_test = pd.read_csv(TRIP5_POTHOLES_PATH)
    raw_sensors_test = pd.read_csv(TRIP5_SENSORS_PATH)

    raw_sensors_data1["trip_id"] = 1
    raw_sensors_data2["trip_id"] = 2
    raw_sensors_data3["trip_id"] = 3
    raw_sensors_data4["trip_id"] = 4
    raw_sensors_test["trip_id"] = 5

    raw_potholes_data1["trip_id"] = 1
    raw_potholes_data2["trip_id"] = 2
    raw_potholes_data3["trip_id"] = 3
    raw_potholes_data4["trip_id"] = 4
    raw_potholes_test["trip_id"] = 5

    raw_sensors = pd.concat([
        raw_sensors_data1,
        raw_sensors_data2,
        raw_sensors_data3,
        raw_sensors_data4
    ], ignore_index=True)

    raw_potholes = pd.concat([
        raw_potholes_data1,
        raw_potholes_data2,
        raw_potholes_data3,
        raw_potholes_data4
    ], ignore_index=True)

    raw_sensors = raw_sensors.sort_values(['trip_id', 'timestamp']).reset_index(drop=True)
    raw_sensors_test = raw_sensors_test.sort_values(['timestamp']).reset_index(drop=True)
    return raw_sensors, raw_potholes, raw_sensors_test, raw_potholes_test

def sliding_window(sensor, pothole):
    # Sliding Window 생성 
    # sensor data를 슬라이딩 윈도우로 만들어서 윈도우 내에 포트홀 타임스탬프가 존재하면 포트홀로 분류함
    # -> 전체 데이터 10000개 중 포트홀 타임스탬프는 100개이므로 데이터 불균형때문에 이렇게 설정했음
    windows = []
    labels = []

    for trip_id in sensor["trip_id"].unique():
        sensor_trip = sensor[sensor["trip_id"] == trip_id].reset_index(drop=True)
        pothole_trip = pothole[pothole["trip_id"] == trip_id]

        for i in range(len(sensor_trip) - WINDOW_SIZE):
            window = sensor_trip.iloc[i:i+WINDOW_SIZE]
            # start = window["timestamp"].iloc[0]
            # end = window["timestamp"].iloc[-1]
            center = window["timestamp"].iloc[len(window)//2]
            sampling_interval = window["timestamp"].diff().median()
            delta = sampling_interval * (WINDOW_SIZE // 4)
            delta = np.clip(
                        delta,
                        sampling_interval * 2,
                        sampling_interval * 10
                    )
            # delta = WINDOW_SIZE // 4

            # pothole이 이 구간에 포함되어있는지 확인
            is_pothole = ((pothole_trip["timestamp"] >= center - delta) &
                        (pothole_trip["timestamp"] <= center + delta)).any()

            # is_pothole = ((pothole_trip["timestamp"] >= start) & (pothole_trip["timestamp"] <= end)).any()

            label = 1 if is_pothole else 0

            windows.append(window)
            labels.append(label)

    print(len(windows))
    print(labels.count(1))  # 169 / 2202
    return windows, labels

def featurize(windows):
    features = []

    for window in windows:
        feature = {}

        # 사용할 컬럼
        cols = [
            "accelerometerX", "accelerometerY", "accelerometerZ",
            "gyroX", "gyroY", "gyroZ",
            "speed"
        ]
        acc_mag = np.sqrt(
            window["accelerometerX"]**2 +
            window["accelerometerY"]**2 +
            window["accelerometerZ"]**2
        )

        feature["acc_mag_mean"] = acc_mag.mean()
        feature["acc_mag_max"] = acc_mag.max()
        feature["acc_mag_std"] = acc_mag.std()

        for col in cols:
            values = window[col]

            feature[f"{col}_mean"] = values.mean()
            feature[f"{col}_std"] = values.std()
            feature[f"{col}_min"] = values.min()
            feature[f"{col}_max"] = values.max()

            feature[f"{col}_range"] = values.max() - values.min()

            # 변화량 (충격 감지 핵심)
            diff = values.diff().fillna(0)
            feature[f"{col}_diff_mean"] = diff.mean()
            feature[f"{col}_diff_max"] = diff.max()

        features.append(feature)

    return features

def make_dataset(sensor, pothole):
    print("SLIDING WINDOW 생성")
    windows, labels = sliding_window(sensor, pothole)

    print("윈도우별 FEATURE 추출")
    features = featurize(windows)

    print("DataFrame 생성")
    data = pd.DataFrame(features)
    data['label'] = labels
    data = data.dropna()

    return data

def data_oversampling(data):
    from imblearn.over_sampling import SMOTE

    x = data.drop('label', axis=1)
    y = data['label']

    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=3,
        random_state=42
    )

    X_res, y_res = smote.fit_resample(x, y)

    resampled = pd.concat([X_res, y_res], axis=1)
    resampled = resampled[data.columns]
    return resampled

def data_downsampling(data):
    from sklearn.utils import resample

    df_majority = data[data['label'] == 0]
    df_minority = data[data['label'] == 1]

    # minority 개수에 맞춤
    df_majority_down = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority_down, df_minority])

    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def clip_outliers(data, Q1=None, Q3=None):
    features = data.drop(columns=['label'])

    if Q1 is None or Q3 is None:
        Q1 = features.quantile(0.25)
        Q3 = features.quantile(0.75)
    IQR = Q3 - Q1

    features = features.clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR, axis=1)

    data = pd.concat([features, data['label']], axis=1)

    return data, Q1, Q3

def main():
    print('1. RAW DATA LOADING ...')
    raw_sensors, raw_potholes, raw_sensors_test, raw_potholes_test = load_data()
    print(raw_sensors[:3])

    print("2. Train 데이터 생성")
    train_data = make_dataset(raw_sensors, raw_potholes)

    # outlier clipping
    train_data, q1, q3 = clip_outliers(train_data)
    # SMOTE 오버샘플링
    # train_data = data_oversampling(train_data)
    # 다운샘플링
    train_data = data_downsampling(train_data)

    print("3. Test 데이터 생성")
    test_data = make_dataset(raw_sensors_test, raw_potholes_test)
    test_data, _, _ = clip_outliers(test_data, q1, q3)

    print("\nTrain label 분포")
    print(train_data["label"].value_counts())

    print("\nTest label 분포")
    print(test_data["label"].value_counts())

    save_path = os.path.join(project_root, "data", "processed", "pothole")
    os.makedirs(save_path, exist_ok=True)

    print("\n5. CSV 저장")

    train_data.to_csv(os.path.join(save_path, "train_raw_clip_down.csv"), index=False)
    test_data.to_csv(os.path.join(save_path, "test_raw_clip_down.csv"), index=False)

    print("저장 완료:", save_path)

if __name__=='__main__':
    main()