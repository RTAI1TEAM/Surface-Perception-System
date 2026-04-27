import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pothole', 'train_v3_s2.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'outdoor_eda')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_all(data, windows=None, threshold=None, save=True):
    print("\n===== 통합 시각화 시작 =====")

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
        "az_max", "az_std", "acc_mag_mean",
        "gz_zc", "speed_mean"
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
    for col in features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="label", y=col, data=data)
        plt.title(f"{col} by label")
        if save:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_boxplot.png"))
        plt.show()

    # -------------------------------
    # 4. 상관관계 히트맵 (스타일 통일)
    # -------------------------------
    # 주요 피처만 선택하여 가독성 확보
    cols_to_plot = [c for c in data.columns if ('mean' in c or 'std' in c or 'max' in c) and ('a' in c or 'speed' in c or 'mag' in c)]
    if not cols_to_plot:
        cols_to_plot = data.select_dtypes(include=[np.number]).columns[:12]
    
    corr = data[cols_to_plot].corr()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, annot_kws={"size": 7}, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=.5, cbar=False)
    plt.title('Feature Correlation (Outdoor)', fontsize=12)
    plt.tick_params(labelsize=8)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
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
    print("6. 속도 vs 가속도 산점도 생성 중...")
    plt.figure(figsize=(8, 6))
    
    # 산점도 그리기 (정상: 파란색, 포트홀: 빨간색)
    sns.scatterplot(
        data=data, 
        x='speed_mean', 
        y='az_std', 
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

    print("===== 시각화 완료 =====\n")


# -------------------------------
# 실행
# -------------------------------
if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)

    visualize_all(data, save=True)