import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# 모델들 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "indoor", "indoor_train_features.csv")
# 요청하신 대로 성능 수치는 performance 폴더에 저장하도록 경로 유지
REPORT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance")
PLOT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance") # 일관성을 위해 수정

def load_processed_data(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
    df = pd.read_csv(file_path)
    target_col = 'surface_encoded'
    y = df[target_col]
    X = df.drop(columns=['series_id', 'group_id', 'surface', target_col])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def run_experiment():
    X_train, X_test, y_train, y_test = load_processed_data()
    
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"🚀 Testing {name}...")
        
        # [1. Baseline] - 원본 데이터 학습
        model.fit(X_train, y_train)
        f1_base = f1_score(y_test, model.predict(X_test), average='macro')
        
        # [2. Class Weight] - 모델 가중치 부여
        f1_weight = 0
        try:
            if name == "XGBoost":
                s_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                model.fit(X_train, y_train, sample_weight=s_weights)
                f1_weight = f1_score(y_test, model.predict(X_test), average='macro')
            elif hasattr(model, 'class_weight'):
                model.set_params(class_weight='balanced')
                model.fit(X_train, y_train)
                f1_weight = f1_score(y_test, model.predict(X_test), average='macro')
                model.set_params(class_weight=None) # 원상복구
        except Exception as e:
            f1_weight = 0

        # [3. SMOTE] - 오버샘플링
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model.fit(X_res, y_res)
        f1_smote = f1_score(y_test, model.predict(X_test), average='macro')

        results.append({
            "Model": name,
            "Baseline": f1_base,
            "ClassWeight": f1_weight,
            "SMOTE": f1_smote
        })

    # 데이터 정리
    df_res = pd.DataFrame(results)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(REPORT_DIR, "indoor_experiment_results.csv"), index=False)

    # 시각화 설정
    df_plot = df_res.melt(id_vars='Model', var_name='Strategy', value_name='F1_Score')
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # 막대 그래프 생성
    ax = sns.barplot(data=df_plot, x='Model', y='F1_Score', hue='Strategy', palette='viridis')
    
    # 수치 표시 (Annotation)
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # 0점인 경우(NB 가중치 등) 표시 생략
            ax.annotate(f'{height:.3f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=10, fontweight='bold')

    plt.title('Performance Comparison: Baseline vs. Imbalance Handling', fontsize=18, pad=20)
    plt.ylabel('F1-Score (Macro)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1.15) # 수치 표시 공간 확보를 위해 상단 여백 추가
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=0) # 모델명이 잘 보이도록 0도로 수정
    
    # 그래프 저장 및 출력
    plot_path = os.path.join(REPORT_DIR, "experiment_strategy_comparison_with_values.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ 모든 실험 및 시각화 완료! 저장 위치: {plot_path}")

if __name__ == "__main__":
    run_experiment()