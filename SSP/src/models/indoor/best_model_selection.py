import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

def visualize_3_stage_evolution():
    # 2. 데이터 로드
    print("🚀 최종 성능 비교를 위한 데이터 로드 중...")
    _, X_test, _, y_test = load_processed_data()

    # [중요] 각 단계별 파일명이 실제 폴더에 있는 이름과 정확히 일치해야 합니다.
    evolution_map = {
        "DecisionTree": [
            "Base_Decision_Tree.pkl", 
            "Best_DecisionTree_ClassWeight.pkl", 
            "Final_Best_DecisionTree.pkl"
        ],
        "RandomForest": [
            "Base_Random_Forest.pkl", 
            "Best_RandomForest_SMOTE.pkl", 
            "Final_Best_RandomForest.pkl"
        ],
        "XGBoost": [
            "Base_XGBoost.pkl", 
            "Best_XGBoost_Baseline.pkl", 
            "Final_Best_XGBoost.pkl"
        ]
    }

    evolution_results = []
    stages = ["1.Baseline", "2.Imbalance_Handled", "3.Hyperparameter_Tuned"]

    # 3. 데이터 수집
    for model_name, files in evolution_map.items():
        for stage, file_name in zip(stages, files):
            path = os.path.join(MODEL_DIR, file_name)
            if os.path.exists(path):
                model = joblib.load(path)
                y_pred = model.predict(X_test)
                
                score = f1_score(y_test, y_pred, average='macro')
                
                evolution_results.append({
                    "Model": model_name,
                    "Stage": stage,
                    "F1_Score": score
                })
            else:
                print(f"⚠️ 파일을 찾을 수 없음: {file_name}")

    df_ev = pd.DataFrame(evolution_results)

    # 4. 시각화 (배경 Barplot 제거, 오직 선 그래프만)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # 선 그래프 그리기
    ax = sns.pointplot(
        data=df_ev, 
        x='Stage', 
        y='F1_Score', 
        hue='Model',
        markers=["o", "s", "D"], 
        linestyles=["-", "--", "-."],
        palette="deep",
        scale=1.2 # 점과 선의 굵기 조절
    )

    # 각 점 위에 수치 표시
    for i in range(len(df_ev)):
        row = df_ev.iloc[i]
        # x축 인덱스 찾기
        stage_idx = stages.index(row['Stage'])
        ax.text(
            stage_idx, 
            row['F1_Score'] + 0.005, 
            f"{row['F1_Score']:.3f}", 
            ha='center', 
            va='bottom', 
            fontweight='bold', 
            fontsize=11
        )

    plt.title('Final Model Evolution: 3-Step Improvement', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Optimization Stage', fontsize=12)
    plt.ylim(df_ev['F1_Score'].min() - 0.05, 1.0) # 수치에 맞춰 y축 하한값 자동 조절
    plt.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 이미지 저장
    save_path = os.path.join(REPORT_DIR, "final_3step_evolution.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 시각화 완료! 저장 위치: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    visualize_3_stage_evolution()