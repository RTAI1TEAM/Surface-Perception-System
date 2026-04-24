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
        available = df['surface'].unique()[:5]

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    for ax, surface, color in zip(axes, available, colors):
        # 해당 재질의 첫 번째 series_id 사용
        sid = df[df['surface'] == surface]['series_id'].iloc[0]
        series = df[df['series_id'] == sid]

        ax.plot(series['measurement_number'],
                series['linear_acceleration_Z'],
                color=color, linewidth=1.5)
        ax.set_ylabel('Linear Acc Z\n(m/s²)', fontsize=9)
        ax.set_title(f'Surface: {surface}  (series_id={sid})', fontsize=10, loc='left')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    axes[-1].set_xlabel('Measurement Number (0 ~ 127)', fontsize=10)
    fig.suptitle('[Fig 3-1] Raw Linear Acceleration Z — Per Surface Type\n(Before Preprocessing)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_raw_timeseries_per_surface.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── 2. 원본 쿼터니언 축 간 상관관계 (변환 필요성 입증) ────────────────────
def plot_raw_quaternion_corr(df: pd.DataFrame) -> None:
    """
    원본 쿼터니언 (X, Y, Z, W) 4개 축 간의 상관관계를 시각화하여
    다중공선성 문제가 실재함을 보인다.
    """
    quat_cols = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']
    sample = df[df['series_id'] == df['series_id'].iloc[0]][quat_cols]

    # 각 series 단위로 평균을 내어 sample 상관 계산
    agg = df.groupby('series_id')[quat_cols].mean()
    corr = agg.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 좌: 히트맵
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, ax=axes[0])
    axes[0].set_title('Quaternion Axis Correlation\n(mean per series_id)', fontsize=11)

    axes[1].scatter(agg['orientation_X'], agg['orientation_W'],
                    alpha=0.4, s=15, color='steelblue')
    axes[1].set_xlabel('orientation_X')
    axes[1].set_ylabel('orientation_W')
    axes[1].set_title(f'orientation_X vs W Scatter\n(r = {corr.loc["orientation_X","orientation_W"]:.2f})', fontsize=11)

    fig.suptitle('[Fig 3-2] Quaternion Multicollinearity (Before Preprocessing)',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_raw_quaternion_corr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── 3. 원본 가속도 분포 (재질별 raw 값 중첩 히스토그램) ───────────────────
def plot_raw_accel_distribution(df: pd.DataFrame) -> None:
    """
    전처리 전 원본 linear_acceleration_Z 값이
    재질별로 얼마나 구분되는지(또는 혼재되는지)를 보인다.
    단순 mean만으로는 분류가 어렵다는 근거가 된다.
    """
    surfaces = df['surface'].unique()
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(surfaces)))

    fig, ax = plt.subplots(figsize=(11, 5))
    for surface, color in zip(surfaces, colors):
        vals = df[df['surface'] == surface]['linear_acceleration_Z']
        ax.hist(vals, bins=80, alpha=0.45, label=surface, color=color, density=True)

    ax.set_xlabel('linear_acceleration_Z (m/s²)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('[Fig 3-3] Raw Linear Acceleration Z Distribution — Per Surface Type\n'
                 '(Distributions overlap: simple mean is insufficient for classification)', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
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
