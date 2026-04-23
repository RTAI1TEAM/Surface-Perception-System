import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

from data.routes_indoor import load_processed_data

def evaluate_and_visualize():
    # 1. 테스트 데이터 로드
    print("🚀 테스트 데이터를 로드 중입니다...")
    _, X_test, _, y_test = load_processed_data()
    
    # 2. 모델 저장 경로 설정 (SSP/models/indoor/)
    model_dir = os.path.join(os.path.dirname(project_root), "models", "indoor")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ 로드할 모델 파일이 없습니다.")
        return

    results_metrics = {}
    cms = {}
    model_names = []

    # 3. 모델 평가
    for f in model_files:
        name = f.replace('.pkl', '').replace('_', ' ')
        model_names.append(name)
        model = joblib.load(os.path.join(model_dir, f))
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        results_metrics[name] = [acc, prec, rec, f1]
        cms[name] = confusion_matrix(y_test, y_pred)
        print(f"✅ {name} 평가 완료")

    # 4. 동적 시각화 (모델당 2개의 세로 칸 할당)
    n_models = len(model_names)
    # 모델당 2줄(CM, Metrics)씩 사용하므로 n_models 행, 2열 구조로 배치
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 5 * n_models))
    
    # 모델이 1개일 경우를 대비해 2차원 배열 유지
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    metric_labels = ['Acc', 'Prec', 'Rec', 'F1']

    for i, name in enumerate(model_names):
        # 왼쪽 열: Confusion Matrix
        sns.heatmap(cms[name], annot=True, fmt='d', cmap='Blues', ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title(f"CM: {name}", fontsize=12, fontweight='bold')
        
        # 오른쪽 열: 4대 지표
        scores = results_metrics[name]
        sns.barplot(x=metric_labels, y=scores, ax=axes[i, 1], palette='viridis')
        axes[i, 1].set_title(f"Performance: {name}", fontsize=12, fontweight='bold')
        axes[i, 1].set_ylim(0, 1.1)
        for j, v in enumerate(scores):
            axes[i, 1].text(j, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.suptitle("Factory-Twin Guard: Multi-Model Evaluation", fontsize=20, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    output_img = os.path.join(project_root, "evaluation_summary.png")
    plt.savefig(output_img)
    print(f"\n📊 시각화 완료: {output_img}")
    plt.show()

if __name__ == "__main__":
    evaluate_and_visualize()