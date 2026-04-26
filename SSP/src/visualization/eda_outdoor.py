import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pothole', 'train.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'outdoor_eda')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_all(data, windows=None, threshold=None, save=True):
    print("\n📊 ===== 통합 시각화 시작 =====")

    # -------------------------------
    # 1. Label 분포
    # -------------------------------
    plt.figure(figsize=(5,4))
    data["label"].value_counts().plot(kind="bar")
    plt.title("Label Distribution")
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"))
    plt.show()

    # -------------------------------
    # 2. Feature 분포 비교
    # -------------------------------
    features = [
        "acc_z_max", "acc_z_std", "acc_mean",
        "speed_mean", "speed_std",
        "peak_count", "acc_diff_mean", "acc_diff_max"
    ]

    for col in features:
        plt.figure(figsize=(6,4))

        plt.hist(data[data["label"]==0][col], bins=50, alpha=0.5, label="normal")
        plt.hist(data[data["label"]==1][col], bins=50, alpha=0.5, label="pothole")

        plt.title(f"{col} distribution")
        plt.legend()

        if save:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_distribution.png"))
        plt.show()

    # -------------------------------
    # 3. boxplot (label 비교)
    # -------------------------------
    for col in ["acc_z_max", "acc_z_std", "peak_count", "acc_diff_max"]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="label", y=col, data=data)
        plt.title(f"{col} by label")
        if save:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_boxplot.png"))
        plt.show()

    # -------------------------------
    # 4. 상관관계 히트맵
    # -------------------------------
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.show()

    # -------------------------------
    # 5. Peak 시각화 (옵션)
    # -------------------------------
    if windows is not None and threshold is not None:
        sample_window = windows[0]
        acc_z = sample_window["accelerometerZ"].values

        peaks, _ = find_peaks(acc_z, height=threshold)

        plt.figure(figsize=(10,4))
        plt.plot(acc_z, label="acc_z")
        plt.scatter(peaks, acc_z[peaks], color="red", label="peaks")
        plt.axhline(threshold, color="green", linestyle="--", label="threshold")

        plt.legend()
        plt.title("Peak Detection Example")
        if save:
            plt.savefig(os.path.join(OUTPUT_DIR, "peak_detection.png"))
        plt.show()

    # -------------------------------
    # 6. 속도 vs 가속도 산점도 (비선형성 확인)
    # -------------------------------
    print("▶ 6. 속도 vs 가속도 산점도 생성 중...")
    plt.figure(figsize=(8, 6))
    
    # 산점도 그리기 (정상: 파란색, 포트홀: 빨간색)
    sns.scatterplot(
        data=data, 
        x='speed_mean', 
        y='acc_z_std', 
        hue='label', 
        palette={0: '#1f77b4', 1: '#d62728'}, 
        alpha=0.4
    )

    plt.title("Non-linear Relationship: Speed vs Acc_Z_Std")
    plt.xlabel("Average Speed (m/s)")
    plt.ylabel("Z-Acceleration Std Dev (g)")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "speed_vs_acc_z_std_scatter.png"))
    plt.show()

    print("📊 ===== 시각화 완료 =====\n")


# -------------------------------
# 실행
# -------------------------------
if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)

    visualize_all(data, save=True)