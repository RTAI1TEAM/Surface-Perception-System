import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 6개 모델 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.routes_indoor import load_processed_data

def run_full_tuning_viz():
    # 1. 데이터 로드
    X_train, X_test, y_train, y_test = load_processed_data()

    # 2. 6개 모델 및 튜닝 파라미터 정의
    model_configs = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {'C': [0.1, 1, 10], 'kernel': ['rbf']}
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {'var_smoothing': np.logspace(0, -9, num=5)}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {'n_estimators': [100, 200], 'max_depth': [10, None]}
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "params": {'C': [0.1, 1, 10]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='mlogloss', random_state=42),
            "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
        }
    }

    final_results = []

    # 3. 튜닝 및 결과 수집
    for name, config in model_configs.items():
        print(f"🔍 {name} 튜닝 중...")
        
        # Before Tuning
        base_model = config['model'].fit(X_train, y_train)
        before_acc = accuracy_score(y_test, base_model.predict(X_test))
        
        # After Tuning (GridSearchCV)
        grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        after_acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test))
        
        final_results.append({
            'Model': name,
            'Before Tuning': before_acc,
            'After Tuning': after_acc
        })

    # 4. 데이터프레임 변환
    df_plot = pd.DataFrame(final_results)

    # 5. 6개 모델 전체 시각화
    plt.figure(figsize=(15, 8))
    
    # 막대 위치 설정
    bar_width = 0.35
    index = np.arange(len(df_plot))

    # Before(회색) / After(하늘색) 막대 그리기
    rects1 = plt.bar(index, df_plot['Before Tuning'], bar_width, label='Before Tuning', color='#BDC3C7')
    rects2 = plt.bar(index + bar_width, df_plot['After Tuning'], bar_width, label='After Tuning', color='#5DADE2')

    plt.xlabel('Machine Learning Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('6 Models: Hyperparameter Tuning Comparison (Before vs After)', fontsize=16, fontweight='bold')
    plt.xticks(index + bar_width/2, df_plot['Model'], rotation=15)
    plt.legend()

    # 정확도 수치 표시
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Y축 범위 조정 (최소값 근처에서 시작하게)
    plt.ylim(df_plot[['Before Tuning', 'After Tuning']].values.min() - 0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_tuning_viz()