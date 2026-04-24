import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
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

def run_final_tuning_and_save():
    print("🚀 데이터 로드 및 튜닝 준비 중...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # 튜닝 설정 (Best 모델 파일명과 매칭)
    tuning_configs = {
        "DecisionTree": {
            "source_file": "Best_DecisionTree_ClassWeight.pkl",
            "save_file": "Final_Tuned_DecisionTree.pkl",
            "strategy": "ClassWeight",
            "params": {
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10]
            }
        },
        "RandomForest": {
            "source_file": "Best_RandomForest_SMOTE.pkl",
            "save_file": "Final_Tuned_RandomForest.pkl",
            "strategy": "SMOTE",
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "min_samples_leaf": [1, 2]
            }
        },
        "XGBoost": {
            "source_file": "Best_XGBoost_ClassWeight.pkl",
            "save_file": "Final_Tuned_XGBoost.pkl",
            "strategy": "ClassWeight",
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }
    }

    for name, config in tuning_configs.items():
        source_path = os.path.join(MODEL_DIR, config['source_file'])
        if not os.path.exists(source_path):
            print(f"⚠️ 원본 파일을 찾을 수 없습니다: {config['source_file']}")
            continue

        print(f"\n🔥 {name} 하이퍼파라미터 튜닝 중...")
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

        # GridSearchCV 수행
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=config['params'],
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid.fit(current_X, current_y, **fit_params)

        # 최적 모델 추출 및 저장
        best_tuned_model = grid.best_estimator_
        save_path = os.path.join(MODEL_DIR, config['save_file'])
        joblib.dump(best_tuned_model, save_path)
        
        # 성능 확인
        score = f1_score(y_test, best_tuned_model.predict(X_test), average='weighted')
        print(f"✅ {name} 튜닝 완료! (F1: {score:.4f})")
        print(f"💾 저장 경로: {save_path}")

    print("\n✨ 모든 최종 모델이 성공적으로 저장되었습니다. 이제 시각화 코드를 실행하세요!")

if __name__ == "__main__":
    run_final_tuning_and_save()