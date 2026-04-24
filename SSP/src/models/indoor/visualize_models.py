import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)

from data.routes_indoor import load_processed_data

def load_and_visualize():
    model_dir = r"C:\work\mlproject\Surface-Perception-System\SSP\models\indoor"
    figure_save_path = r"C:\work\mlproject\Surface-Perception-System\SSP\reports\figures\indoor_performance"
    os.makedirs(figure_save_path, exist_ok=True)

    print("🚀 테스트 데이터를 불러오는 중...")
    _, X_test, _, y_test = load_processed_data()

    # 불러올 모델 파일 리스트 (Base_로 시작하는 pkl)
    model_files = [f for f in os.listdir(model_dir) if f.startswith('Base_') and f.endswith('.pkl')]
    
    if not model_files:
        print("❌ 저장된 모델 파일이 없습니다. 먼저 학습을 진행해 주세요."); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model-Specific Performance (Loaded from PKL)', fontsize=22, fontweight='bold')
    axes = axes.flatten()

    print("\n--- 모델 로드 및 성능 평가 시작 ---")
    for i, f_name in enumerate(model_files[:6]): # 최대 6개 모델
        model = joblib.load(os.path.join(model_dir, f_name))
        y_pred = model.predict(X_test)
        
        # 지표 산출
        metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='macro'),
                recall_score(y_test, y_pred, average='macro'),
                f1_score(y_test, y_pred, average='macro')
            ]
        }
        m_df = pd.DataFrame(metrics)
        
        # 2x3 서브플롯 시각화
        display_name = f_name.replace('Base_', '').replace('.pkl', '').replace('_', ' ')
        sns.barplot(x='Metric', y='Score', data=m_df, ax=axes[i], palette='coolwarm')
        axes[i].set_title(f"Model: {display_name}", fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 이미지 저장
    save_path = os.path.join(figure_save_path, "loaded_model_metrics.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 시각화 파일 저장 완료: {save_path}")
    plt.show()

if __name__ == "__main__":
    load_and_visualize()