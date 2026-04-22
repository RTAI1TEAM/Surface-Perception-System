import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import koreanize_matplotlib

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
    WINDOW_SIZE = 15
    windows = []
    labels = []

    for trip_id in sensor["trip_id"].unique():
        sensor_trip = sensor[sensor["trip_id"] == trip_id].reset_index(drop=True)
        pothole_trip = pothole[pothole["trip_id"] == trip_id]

        for i in range(len(sensor_trip) - WINDOW_SIZE):
            window = sensor_trip.iloc[i:i+WINDOW_SIZE]
            start = window["timestamp"].iloc[0]
            end = window["timestamp"].iloc[-1]

            # pothole이 이 구간에 포함되어있는지 확인
            is_pothole = ((pothole_trip["timestamp"] >= start) & (pothole_trip["timestamp"] <= end)).any()

            label = 1 if is_pothole else 0

            windows.append(window)
            labels.append(label)

    print(len(windows))
    print(labels.count(1))  # 169 / 2202
    return windows, labels

# window 별로 최대 충격, 흔들림 정도, 전체 진동 강도, 속도, 회전율, peak 등을 구함
def featurize(windows, threshold):
    features = []
    
    
    # 윈도우별 z축 가속도값 통계
    for window in windows:
        acc_z = window["accelerometerZ"]

        peaks, _ = find_peaks(acc_z, height=threshold)
        
        accel_mag = np.sqrt(
                window["accelerometerX"]**2 +
                window["accelerometerY"]**2 +
                window["accelerometerZ"]**2
            )

        # 변화량
        accel_diff = accel_mag.diff().fillna(0)

        feature = { 
            "acc_z_max": acc_z.max(),   # 최대 충격 크기
            "acc_z_std": acc_z.std(),   # 흔들림 정도
            "acc_mean": accel_mag.mean(), 
            # 벡터 크기 평균 -> 전체 진동 강도
            "speed_mean": window["speed"].mean(),   # 속도 정보
            "speed_std": window["speed"].std(),   # 
            "gyro_std": window["gyroX"].std(),   # 회전 변화
            "peak_count": len(peaks),
            "acc_diff_mean": accel_diff.mean(),
            "acc_diff_max":accel_diff.max()
        }
        
        features.append(feature)

    return features

def make_dataset(sensor, pothole, threshold):
    print("SLIDING WINDOW 생성")
    windows, labels = sliding_window(sensor, pothole)

    print("Feature 생성")
    features = featurize(windows, threshold)

    print("DataFrame 생성")
    data = pd.DataFrame(features)
    data['label'] = labels

    data = data.dropna()

    return data

def plot_data(sensor, pothole):
    fig, axes = plt.subplots(1, 3, figsize=(18,4))

    # 1. 히스토그램
    axes[0].hist(sensor["accelerometerZ"], bins=100)
    axes[0].set_title("acc_z distribution")

    # 2. 박스플롯
    axes[1].boxplot(sensor["accelerometerZ"])
    axes[1].set_title("acc_z boxplot")

    # 3. 시계열 + 포트홀
    axes[2].plot(sensor["accelerometerZ"], label="acc_z")

    pothole_indices = sensor[sensor["timestamp"].isin(pothole["timestamp"])].index

    axes[2].scatter(
        pothole_indices,
        sensor.loc[pothole_indices, "accelerometerZ"],
        color="red",
        label="pothole"
    )

    axes[2].legend()
    axes[2].set_title("acc_z with pothole")

    plt.show()

def main():
    print('1. RAW DATA LOADING ...')
    raw_sensors, raw_potholes, raw_sensors_test, raw_potholes_test = load_data()
    print(raw_sensors[:3])

    print("2. EDA 분석")
    plot_data(raw_sensors, raw_potholes)

    # threshold 계산
    global_threshold = raw_sensors["accelerometerZ"].quantile(0.95)

    print("3. Train 데이터 생성")
    train_data = make_dataset(raw_sensors, raw_potholes, global_threshold)
    
    print("4. Test 데이터 생성")
    test_data = make_dataset(raw_sensors_test, raw_potholes_test, global_threshold)

    print("\nTrain label 분포")
    print(train_data["label"].value_counts())

    print("\nTest label 분포")
    print(test_data["label"].value_counts())

    save_path = os.path.join(project_root, "data", "processed", "pothole")
    os.makedirs(save_path, exist_ok=True)

    print("\n5. CSV 저장")

    train_data.to_csv(os.path.join(save_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(save_path, "test.csv"), index=False)

    print("저장 완료:", save_path)


if __name__=='__main__':
    main()