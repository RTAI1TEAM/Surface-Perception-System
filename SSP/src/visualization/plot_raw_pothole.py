import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SENSORS_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'pothole', 'trip1_sensors.csv')
POTHOLES_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'pothole', 'trip1_potholes.csv')

import pandas as pd
import matplotlib.pyplot as plt

def plot_all_sensor_with_scatter(sensor_path, pothole_path):
    sensor = pd.read_csv(sensor_path)
    pothole = pd.read_csv(pothole_path)

    sensor = sensor.sort_values("timestamp").reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # =====================
    # 1️⃣ Accelerometer
    # =====================
    axes[0].plot(sensor["timestamp"], sensor["accelerometerX"], label="acc_x")
    axes[0].plot(sensor["timestamp"], sensor["accelerometerY"], label="acc_y")
    axes[0].plot(sensor["timestamp"], sensor["accelerometerZ"], label="acc_z")

    # =====================
    # 2️⃣ Gyro
    # =====================
    axes[1].plot(sensor["timestamp"], sensor["gyroX"], label="gyro_x")
    axes[1].plot(sensor["timestamp"], sensor["gyroY"], label="gyro_y")
    axes[1].plot(sensor["timestamp"], sensor["gyroZ"], label="gyro_z")

    # =====================
    # 3️⃣ Speed
    # =====================
    axes[2].plot(sensor["timestamp"], sensor["speed"], label="speed", color='black')

    # =====================
    # 🔥 pothole scatter 표시 (핵심)
    # =====================
    pothole_points = []

    for t in pothole["timestamp"]:
        # 가장 가까운 센서 timestamp 찾기
        idx = (sensor["timestamp"] - t).abs().idxmin()
        pothole_points.append(sensor.loc[idx])

    pothole_points = pd.DataFrame(pothole_points)

    # 각 subplot에 scatter 표시
    axes[0].scatter(pothole_points["timestamp"], pothole_points["accelerometerZ"],
                    label="pothole", zorder=5)

    axes[1].scatter(pothole_points["timestamp"], pothole_points["gyroZ"],
                    label="pothole", zorder=5)

    axes[2].scatter(pothole_points["timestamp"], pothole_points["speed"],
                    label="pothole", zorder=5)

    # =====================
    # 마무리
    # =====================
    axes[0].set_title("Accelerometer")
    axes[1].set_title("Gyroscope")
    axes[2].set_title("Speed")

    for ax in axes:
        ax.legend()

    axes[2].set_xlabel("Timestamp")

    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_acc_z_overview(sensor_path, pothole_path):
    sensor = pd.read_csv(sensor_path)
    pothole = pd.read_csv(pothole_path)

    sensor = sensor.sort_values("timestamp")

    plt.figure(figsize=(15, 4))

    plt.plot(sensor["timestamp"], sensor["accelerometerZ"], label="acc_z")

    # pothole 표시
    pothole_points = []
    for t in pothole["timestamp"]:
        idx = (sensor["timestamp"] - t).abs().idxmin()
        pothole_points.append(sensor.loc[idx])

    pothole_points = pd.DataFrame(pothole_points)

    plt.scatter(
        pothole_points["timestamp"],
        pothole_points["accelerometerZ"],
        color='red',
        label='pothole',
        zorder=5
    )

    plt.title("Accelerometer Z (Overview)")
    plt.xlabel("Timestamp")
    plt.ylabel("acc_z")
    plt.legend()
    plt.show()

def plot_pothole_zoom(sensor_path, pothole_path, window=1.0, max_plots=10):
    sensor = pd.read_csv(sensor_path)
    pothole = pd.read_csv(pothole_path)

    sensor = sensor.sort_values("timestamp")

    for i, t in enumerate(pothole["timestamp"][:max_plots]):
        subset = sensor[
            (sensor["timestamp"] >= t - window) &
            (sensor["timestamp"] <= t + window)
        ]

        plt.figure(figsize=(8, 3))
        plt.plot(subset["timestamp"], subset["accelerometerZ"], label="acc_z")

        plt.axvline(t, color='red', linestyle='--', label="pothole")

        plt.title(f"Pothole Zoom {i+1}")
        plt.xlabel("Timestamp")
        plt.ylabel("acc_z")
        plt.legend()
        plt.show()

def plot_normalized_acc(sensor_path, pothole_path):
    sensor = pd.read_csv(sensor_path)
    pothole = pd.read_csv(pothole_path)

    sensor = sensor.sort_values("timestamp")

    # 정규화
    acc_cols = ["accelerometerX", "accelerometerY", "accelerometerZ"]
    norm = (sensor[acc_cols] - sensor[acc_cols].mean()) / sensor[acc_cols].std()

    plt.figure(figsize=(15, 4))

    plt.plot(sensor["timestamp"], norm["accelerometerX"], label="acc_x")
    plt.plot(sensor["timestamp"], norm["accelerometerY"], label="acc_y")
    plt.plot(sensor["timestamp"], norm["accelerometerZ"], label="acc_z")

    # pothole 표시
    pothole_points = []
    for t in pothole["timestamp"]:
        idx = (sensor["timestamp"] - t).abs().idxmin()
        pothole_points.append(norm.loc[idx])

    pothole_points = pd.DataFrame(pothole_points)

    plt.scatter(
        sensor.loc[pothole_points.index, "timestamp"],
        pothole_points["accelerometerZ"],
        color='red',
        label='pothole',
        zorder=5
    )

    plt.title("Normalized Accelerometer (X/Y/Z)")
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()

# plot_all_sensor_with_scatter(SENSORS_PATH, POTHOLES_DIR)

plot_acc_z_overview(SENSORS_PATH, POTHOLES_DIR)

plot_pothole_zoom(SENSORS_PATH, POTHOLES_DIR)

plot_normalized_acc(SENSORS_PATH, POTHOLES_DIR)