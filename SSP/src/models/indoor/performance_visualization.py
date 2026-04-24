import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, auc

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance")
os.makedirs(REPORT_DIR, exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

# 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def calculate_pr_auc_multi(y_true, y_score_probas, n_classes):
    y_true_bin = pd.get_dummies(y_true).values
    weights = y_true.value_counts(normalize=True).sort_index().values
    pr_aucs = []
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_score_probas[:, i])
        pr_aucs.append(auc(r, p))
    return np.average(pr_aucs, weights=weights)

def generate_individual_reports():
    print("🚀 알고리즘별 개별 리포트 생성 중...")
    _, X_test, _, y_test = load_processed_data()
    n_classes = len(y_test.unique())

    # 모델 맵 (이전과 동일)
    model_map = {
        "DecisionTree": ["Base_Decision_Tree.pkl", "Best_DecisionTree_ClassWeight.pkl", "Final_Best_DecisionTree.pkl"],
        "RandomForest": ["Base_Random_Forest.pkl", "Best_RandomForest_SMOTE.pkl", "Final_Best_RandomForest.pkl"],
        "XGBoost":      ["Base_XGBoost.pkl",      "Best_XGBoost_Baseline.pkl", "Final_Best_XGBoost.pkl"]
    }
    
    stages = ["Base", "Imbalance", "Final"]
    
    # 각 알고리즘별로 반복하며 1장씩 그리기
    for algo, files in model_map.items():
        algo_results = []
        
        for stage, f_name in zip(stages, files):
            path = os.path.join(MODEL_DIR, f_name)
            if not os.path.exists(path): continue
            
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            # 지표 계산
            f1 = f1_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            pr_auc = calculate_pr_auc_multi(y_test, y_proba, n_classes) if y_proba is not None else 0

            algo_results.append({"Stage": stage, "PR-AUC": pr_auc, "Recall": rec, "F1-Score": f1})

        # 데이터 재구성 (Melt)
        df_algo = pd.DataFrame(algo_results).melt(id_vars="Stage", var_name="Metric", value_name="Score")

        # --- 시각화 시작 ---
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid", font='Malgun Gothic')
        
        # 지표별 막대 그래프
        ax = sns.barplot(data=df_algo, x="Stage", y="Score", hue="Metric", palette="viridis")
        
        # 막대 위에 수치 표시
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                            ha='center', va='bottom', xytext=(0, 5), 
                            textcoords='offset points', fontsize=11, fontweight='bold')

        plt.title(f"모델 성능 발전사: {algo}", fontsize=20, fontweight='bold', pad=20)
        plt.ylim(0, 1.15)
        plt.ylabel("Performance Score", fontsize=12)
        plt.xlabel("개선 단계", fontsize=12)
        plt.legend(title="평가 지표", bbox_to_anchor=(1, 1))
        
        # 파일 저장
        save_name = f"report_performance_{algo}.png"
        save_path = os.path.join(REPORT_DIR, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ {algo} 리포트 저장 완료: {save_path}")
        plt.show()
        plt.close()

if __name__ == "__main__":
    generate_individual_reports()