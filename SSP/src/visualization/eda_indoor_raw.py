"""
실내(Indoor) 원본 데이터 시각화 스크립트
- 목적: 전처리 전 raw 시계열 데이터의 특성을 시각화하여 전처리 필요성을 입증
- 출력: reports/figures/indoor_raw/ 폴더
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import seaborn as sns

# Windows 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
X_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'indoor', 'career-con-2019', 'X_train.csv')
Y_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'indoor', 'career-con-2019', 'y_train.csv')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'indoor_raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)
    df = X.merge(y, on='series_id')
    print(f"원본 데이터 로드 완료: {df.shape}")
    return df


# ─── 1. 바닥 재질별 원본 가속도 시계열 파형 비교 ───────────────────────────
def plot_raw_timeseries(df: pd.DataFrame) -> None:
    """
    각 표면 재질에서 대표 series_id 1개씩 선택하여
    원본 128-step 선형 가속도 Z축 파형을 비교한다.
    """
    surfaces = ['carpet', 'concrete', 'fine_concrete', 'hard_tiles_large_space', 'tiled']
    available = [s for s in surfaces if s in df['surface'].unique()]
    if not available:
        available = df['surface'].unique()[:4] # 가독성을 위해 4개로 제한

    n = len(available)
    # 한 줄 배치를 위해 가로를 줄이고 높이를 5로 고정
    fig, axes = plt.subplots(n, 1, figsize=(6, 5), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    for ax, surface, color in zip(axes, available, colors):
        sid = df[df['surface'] == surface]['series_id'].iloc[0]
        series = df[df['series_id'] == sid]

        ax.plot(series['measurement_number'],
                series['linear_acceleration_Z'],
                color=color, linewidth=1.2)
        ax.set_ylabel('Acc Z', fontsize=8)
        ax.set_title(f'{surface}', fontsize=9, loc='left', pad=2)

    axes[-1].set_xlabel('Step', fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_raw_timeseries_per_surface.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── 2. 원본 쿼터니언 축 간 상관관계 (변환 필요성 입증) ────────────────────
def plot_raw_quaternion_corr(df: pd.DataFrame) -> None:
    """
    원본 쿼터니언 (X, Y, Z, W) 4개 축 간의 상관관계를 시각화한다.
    """
    quat_cols = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']
    agg = df.groupby('series_id')[quat_cols].mean()
    corr = agg.corr()

    # 높이를 5로 고정
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar=False)
    ax.set_title('Quaternion Correlation', fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_raw_quaternion_corr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── 3. 원본 가속도 분포 (재질별 raw 값 중첩 히스토그램) ───────────────────
def plot_raw_accel_distribution(df: pd.DataFrame) -> None:
    """
    전처리 전 원본 linear_acceleration_Z 값이 재질별로 혼재됨을 보인다.
    """
    surfaces = df['surface'].unique()
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(surfaces)))

    # 높이를 5로 고정
    fig, ax = plt.subplots(figsize=(6, 5))
    for surface, color in zip(surfaces, colors):
        vals = df[df['surface'] == surface]['linear_acceleration_Z']
        ax.hist(vals, bins=50, alpha=0.4, label=surface, density=True)

    ax.set_xlabel('Acc Z (m/s²)', fontsize=10)
    ax.set_title('Raw Accel Distribution', fontsize=11)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_raw_accel_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("실내 원본 데이터 시각화 시작")
    print("=" * 50)
    df = load_data()

    print("\n[1/3] 재질별 원본 시계열 파형 생성 중...")
    plot_raw_timeseries(df)

    print("[2/3] 원본 쿼터니언 상관관계 시각화 생성 중...")
    plot_raw_quaternion_corr(df)

    print("[3/3] 원본 가속도 분포 비교 생성 중...")
    plot_raw_accel_distribution(df)

    print("\n완료! 저장 경로:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
