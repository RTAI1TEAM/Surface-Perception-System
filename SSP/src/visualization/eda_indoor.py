import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List # 타입 힌팅 추가

# 폰트 깨짐 방지 및 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'indoor', 'indoor_train_features.csv') 
OUTPUT_DIR_FIG = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'indoor_eda')
OUTPUT_DIR_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'indoor') # 데이터 저장 경로로 변경

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR_FIG, exist_ok=True)
os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)

def load_data() -> pd.DataFrame:
    """전처리된 실내 데이터를 불러오는 함수"""
    print(f"데이터를 불러옵니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 형태: {df.shape}")
    return df

def plot_target_distribution(df: pd.DataFrame) -> None:
    """타겟 변수(surface)의 클래스 분포 시각화"""
    plt.figure(figsize=(10, 6))
    order = df['surface'].value_counts().index
    sns.countplot(data=df, y='surface', order=order, palette='viridis')
    plt.title('노면 재질(Surface)별 샘플 분포', fontsize=16)
    plt.xlabel('샘플 개수', fontsize=12)
    plt.ylabel('노면 재질', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR_FIG, '01_target_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_distributions(df: pd.DataFrame) -> None:
    """주요 센서 통계치 분포 시각화"""
    features_to_plot = [
        'accel_mag_mean', 'gyro_mag_mean', 
        'linear_acceleration_Z_std', 'angular_velocity_Z_std'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        if feature in df.columns:
            sns.boxplot(data=df, x=feature, y='surface', ax=axes[i], palette='Set2')
            axes[i].set_title(f'표면별 {feature} 분포', fontsize=14)
            axes[i].set_xlabel(feature, fontsize=11)
            axes[i].set_ylabel('노면 재질', fontsize=11)
            
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR_FIG, '02_feature_distributions.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """주요 특성 간 상관관계 히트맵 시각화"""
    cols = [c for c in df.columns if ('_mean' in c or '_std' in c) and ('mag' in c or 'diff' in c or 'roll' in c or 'pitch' in c)]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns[:15]
        
    corr = df[cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
    plt.title('주요 센서 특성 간 상관관계', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR_FIG, '03_correlation_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_scatter(df: pd.DataFrame) -> None:
    """PCA 차원 축소 및 2D 산점도 시각화"""
    exclude_cols = ['series_id', 'group_id', 'surface', 'surface_encoded']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]
    
    X = df[feature_cols]
    y = df['surface']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['surface'] = y
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='surface', palette='tab10', alpha=0.7)
    plt.title(f'PCA 차원 축소 (설명 분산 비율: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR_FIG, '04_pca_scatter.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_and_save_thresholds(df: pd.DataFrame) -> None:
    """
    핵심 로직: 재질별 이상 탐지(Anomaly Detection)를 위한 3-Sigma 임계값 추출 및 저장
    """
    # 이상 탐지의 기준이 될 핵심 진동 지표 선택 (예: 가속도 크기의 최대값 및 변화량)
    target_metrics = ['accel_mag_max', 'accel_diff_mean']
    available_metrics = [m for m in target_metrics if m in df.columns]
    
    if not available_metrics:
        print("경고: 임계값 추출을 위한 핵심 컬럼이 존재하지 않습니다.")
        return

    # 재질(surface)별 평균과 표준편차 계산
    stats_df = df.groupby('surface')[available_metrics].agg(['mean', 'std'])
    
    # 3-Sigma 기반 Upper Bound / Lower Bound 계산
    threshold_data = {}
    for metric in available_metrics:
        mean_s = stats_df[(metric, 'mean')]
        std_s = stats_df[(metric, 'std')]
        
        threshold_data[f'{metric}_lower_bound'] = mean_s - (3 * std_s)
        threshold_data[f'{metric}_upper_bound'] = mean_s + (3 * std_s)
        threshold_data[f'{metric}_mean'] = mean_s
        
    threshold_df = pd.DataFrame(threshold_data).reset_index()
    
    # 검증 및 이상 탐지 파이프라인에서 사용할 수 있도록 저장
    save_path = os.path.join(OUTPUT_DIR_DATA, 'indoor_3sigma_thresholds.csv')
    threshold_df.to_csv(save_path, index=False)
    print(f"핵심 로직 처리 완료: 검증용 3-Sigma 임계값 저장 -> {save_path}")

def main() -> None:
    df = load_data()
    if df.empty:
        print("데이터가 비어있습니다. 전처리 파이프라인을 확인하세요.")
        return
        
    print("\n[1/5] 타겟 분포 시각화 생성 중...")
    plot_target_distribution(df)
    
    print("[2/5] 특성 분포 시각화 생성 중...")
    plot_feature_distributions(df)
    
    print("[3/5] 상관관계 히트맵 생성 중...")
    plot_correlation_heatmap(df)
    
    print("[4/5] PCA 차원 축소 시각화 생성 중...")
    plot_pca_scatter(df)
    
    print("[5/5] 재질별 이상 탐지 3-Sigma 임계값 계산 중...")
    calculate_and_save_thresholds(df)
    
    print("\n모든 EDA 시각화 및 통계 추출이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()