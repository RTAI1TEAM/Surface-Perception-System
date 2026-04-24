import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

def run_imbalance_experiment_and_save_best():
    print("🚀 실험 데이터 로드 중...")
    X_train, X_test, y_train, y_test = load_processed_data()
    os.makedirs(REPORT_DIR, exist_ok=True)

    target_models = {
        "DecisionTree": "Base_Decision_Tree.pkl",
        "RandomForest": "Base_Random_Forest.pkl",
        "XGBoost": "Base_XGBoost.pkl"
    }

    results = []

    for name, filename in target_models.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일을 찾을 수 없음: {filename}")
            continue
            
        print(f"🔍 {name} 실험 시작...")
        model_scores = {} # 현재 모델의 전략별 점수 저장용

        # --- [전략 1] Baseline ---
        model = joblib.load(model_path)
        f1_base = f1_score(y_test, model.predict(X_test), average='macro')
        model_scores['Baseline'] = f1_base

        # --- [전략 2] Class Weight ---
        if name == "XGBoost":
            s_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            model.fit(X_train, y_train, sample_weight=s_weights)
        elif hasattr(model, 'class_weight'):
            model.set_params(class_weight='balanced')
            model.fit(X_train, y_train)
        
        f1_weight = f1_score(y_test, model.predict(X_test), average='macro')
        model_scores['ClassWeight'] = f1_weight

        # --- [전략 3] SMOTE ---
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight=None) # 가중치 해제
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
        
        f1_smote = f1_score(y_test, model.predict(X_test), average='macro')
        model_scores['SMOTE'] = f1_smote

        # --- 최적 전략 판별 및 최종 저장 ---
        best_strategy = max(model_scores, key=model_scores.get)
        print(f"🏆 {name} 최적 전략: {best_strategy} (F1: {model_scores[best_strategy]:.4f})")
        
        # 최적 전략으로 모델 재구성
        if best_strategy == 'Baseline':
            final_model = joblib.load(model_path)
        elif best_strategy == 'ClassWeight':
            final_model = joblib.load(model_path)
            if name == "XGBoost":
                final_model.fit(X_train, y_train, sample_weight=s_weights)
            else:
                final_model.set_params(class_weight='balanced')
                final_model.fit(X_train, y_train)
        else: # SMOTE
            final_model = joblib.load(model_path)
            if hasattr(final_model, 'class_weight'):
                final_model.set_params(class_weight=None)
            final_model.fit(X_res, y_res)

        # 최종 모델 저장 (Best_ 접두사 사용)
        final_save_path = os.path.join(MODEL_DIR, f"Best_{name}_{best_strategy}.pkl")
        joblib.dump(final_model, final_save_path)
        
        results.append({
            "Model": name,
            "Baseline": f1_base,
            "ClassWeight": f1_weight,
            "SMOTE": f1_smote
        })

    # 4. 시각화 (평가지표 Macro로 통일)
    df_res = pd.DataFrame(results)
    df_plot = df_res.melt(id_vars='Model', var_name='Strategy', value_name='F1_Score')
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_plot, x='Model', y='F1_Score', hue='Strategy', palette='viridis')
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width()/2., h), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Performance Comparison', fontsize=18, pad=20)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1.15)
    plt.savefig(os.path.join(REPORT_DIR, "imbalance_strategy_comparison.png"), bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_imbalance_experiment_and_save_best()