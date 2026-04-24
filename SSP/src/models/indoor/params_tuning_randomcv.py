import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV # RandomizedSearchCV로 변경
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

def run_randomized_tuning_and_save():
    print("🚀 데이터 로드 및 RandomizedSearch 준비 중...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # 튜닝 설정 (Best 모델 파일명과 매칭)
    tuning_configs = {
        "DecisionTree": {
            "source_file": "Best_DecisionTree_ClassWeight.pkl",
            "save_file": "Random_Tuned_DecisionTree.pkl",
            "strategy": "ClassWeight",
            "params": {
                "max_depth": [5, 10, 15, 20, 25],
                "min_samples_split": [2, 5, 10, 20],
                "criterion": ["gini", "entropy"]
            }
        },
        "RandomForest": {
            "source_file": "Best_RandomForest_SMOTE.pkl",
            "save_file": "Random_Tuned_RandomForest.pkl",
            "strategy": "SMOTE",
            "params": {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            }
        },
        "XGBoost": {
            "source_file": "Best_XGBoost_Baseline.pkl",
            "save_file": "Random_Tuned_XGBoost.pkl",
            "strategy": "Baseline",
            "params": {
                "n_estimators": [100, 300, 500, 700],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "gamma": [0, 0.1, 0.2]
            }
        }
    }

    for name, config in tuning_configs.items():
        source_path = os.path.join(MODEL_DIR, config['source_file'])
        if not os.path.exists(source_path):
            print(f"⚠️ 원본 파일을 찾을 수 없습니다: {config['source_file']}")
            continue

        print(f"\n🎲 {name} RandomizedSearch 시작...")
        base_model = joblib.load(source_path)
        
        # 전략별 데이터셋 구성
        current_X, current_y = X_train, y_train
        fit_params = {}

        if config['strategy'] == "SMOTE":
            smote = SMOTE(random_state=42, k_neighbors=3)
            current_X, current_y = smote.fit_resample(X_train, y_train)
        elif config['strategy'] == "ClassWeight":
            if name == "XGBoost":
                s_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                fit_params = {'sample_weight': s_weights}
            else:
                base_model.set_params(class_weight='balanced')

        # RandomizedSearchCV 수행
        # n_iter=10: 전체 조합 중 무작위로 10개만 골라서 테스트함
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=config['params'],
            n_iter=10, 
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42 # 결과 재현성을 위해 고정
        )
        random_search.fit(current_X, current_y, **fit_params)

        # 최적 모델 추출 및 저장
        best_tuned_model = random_search.best_estimator_
        save_path = os.path.join(MODEL_DIR, config['save_file'])
        joblib.dump(best_tuned_model, save_path)
        
        # 성능 확인
        score = f1_score(y_test, best_tuned_model.predict(X_test), average='macro')
        print(f"✨ {name} 최적 파라미터: {random_search.best_params_}")
        print(f"✅ {name} 튜닝 완료! (Macro F1: {score:.4f})")
        print(f"💾 저장 경로: {save_path}")

    print("\n✨ RandomizedSearch 기반 최종 모델 저장이 완료되었습니다!")

if __name__ == "__main__":
    run_randomized_tuning_and_save()