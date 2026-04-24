import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 절대 경로 설정 (Root 설정)
base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

# 데이터 로더 임포트
from data.routes_indoor import load_processed_data

def evaluate_metrics_only():
    # 2. 데이터 로드 (절대 경로 지정)
    data_file_path = os.path.abspath(os.path.join(base_path, "../data/processed/indoor/indoor_train_features.csv"))
    print(f"🚀 테스트 데이터를 로드 중: {data_file_path}")
    
    # load_processed_data가 경로를 인자로 받는다고 가정 (안 받으면 인자 삭제)
    _, X_test, _, y_test = load_processed_data(file_path=data_file_path)
    
    # 3. 모델 폴더 탐색
    model_dir = os.path.abspath(os.path.join(base_path, "../models/indoor"))
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ 로드할 모델 파일이 없습니다.")
        return

    results_metrics = {}
    model_names = []

    # 4. 모델 평가 및 지표 추출
    for f in model_files:
        name = f.replace('.pkl', '').replace('_', ' ').replace('Tuned ', '⭐') # 튜닝 모델 표시
        model_names.append(name)
        model = joblib.load(os.path.join(model_dir, f))
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        results_metrics[name] = [acc, prec, rec, f1]
        print(f"✅ {name} 지표 계산 완료")

    # 5. 그리드 시각화 (모델 개수에 맞춰 자동 행/열 설정)
    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols  # 올림 계산
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    metric_labels = ['Acc', 'Prec', 'Rec', 'F1']
    colors = ['#3498DB', '#1ABC9C', '#F1C40F', '#E74C3C'] # 각 지표별 고유 색상

    for i, name in enumerate(model_names):
        scores = results_metrics[name]
        sns.barplot(x=metric_labels, y=scores, ax=axes[i], palette=colors, edgecolor='black', alpha=0.8)
        
        axes[i].set_title(f"Model: {name}", fontsize=14, fontweight='bold', pad=15)
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
        # 막대 위에 수치 표시
        for j, v in enumerate(scores):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)

    # 빈 서브플롯 제거
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Indoor Surface Classification: Model Performance Metrics", 
                fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 결과 저장
    output_img = os.path.join(base_path, "model_metrics_comparison.png")
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n📊 지표 분석 시각화 완료: {output_img}")
    plt.show()

if __name__ == "__main__":
    evaluate_metrics_only()