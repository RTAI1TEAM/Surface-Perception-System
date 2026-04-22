import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 폰트 깨짐 방지 및 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'indoor', 'indoor_train_features.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'indoor_eda')

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"데이터를 불러옵니다: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 형태: {df.shape}")
    return df

def plot_target_distribution(df):
    """타겟 변수(surface)의 클래스 분포 시각화"""
    plt.figure(figsize=(10, 6))
    
    # surface별 데이터 개수 시각화
    order = df['surface'].value_counts().index
    sns.countplot(data=df, y='surface', order=order, palette='viridis')
    
    plt.title('노면 재질(Surface)별 샘플 분포', fontsize=16)
    plt.xlabel('샘플 개수', fontsize=12)
    plt.ylabel('노면 재질', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '01_target_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"저장 완료: {save_path}")
    plt.close()

def plot_feature_distributions(df):
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
    save_path = os.path.join(OUTPUT_DIR, '02_feature_distributions.png')
    plt.savefig(save_path, dpi=300)
    print(f"저장 완료: {save_path}")
    plt.close()

def plot_correlation_heatmap(df):
    """주요 특성 간 상관관계 히트맵 시각화"""
    # mean 및 std 특성 일부만 선택 (너무 많으면 시각화가 어려움)
    cols = [c for c in df.columns if ('_mean' in c or '_std' in c) and ('mag' in c or 'diff' in c or 'roll' in c or 'pitch' in c)]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns[:15] # 없으면 숫자형 중 15개
        
    corr = df[cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
    plt.title('주요 센서 특성 간 상관관계', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png')
    plt.savefig(save_path, dpi=300)
    print(f"저장 완료: {save_path}")
    plt.close()

def plot_pca_scatter(df):
    """PCA 차원 축소 및 2D 산점도 시각화"""
    # 수치형 변수만 선택하고 타겟 인코딩 등 무관한 컬럼 제외
    exclude_cols = ['series_id', 'group_id', 'surface', 'surface_encoded']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]
    
    X = df[feature_cols]
    y = df['surface']
    
    # 스케일링 및 PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['surface'] = y
    
    # 시각화
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='surface', palette='tab10', alpha=0.7)
    plt.title(f'PCA 차원 축소 (설명 분산 비율: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '04_pca_scatter.png')
    plt.savefig(save_path, dpi=300)
    print(f"저장 완료: {save_path}")
    plt.close()

def main():
    df = load_data()
    if df.empty:
        print("데이터가 비어있습니다. 전처리 파이프라인을 확인하세요.")
        return
        
    print("\n[1/4] 타겟 분포 시각화 생성 중...")
    plot_target_distribution(df)
    
    print("[2/4] 특성 분포 시각화 생성 중...")
    plot_feature_distributions(df)
    
    print("[3/4] 상관관계 히트맵 생성 중...")
    plot_correlation_heatmap(df)
    
    print("[4/4] PCA 차원 축소 시각화 생성 중...")
    plot_pca_scatter(df)
    
    print("\n모든 EDA 시각화가 성공적으로 완료되었습니다!")
    print(f"결과 확인: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
