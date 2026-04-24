import os
import sys
import joblib
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

def run_bayesian_tuning_and_save():
    print("🚀 데이터 로드 및 Bayesian Optimization(Optuna) 준비 중...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # 튜닝 설정
    tuning_configs = {
        "DecisionTree": {"source": "Best_DecisionTree_ClassWeight.pkl", "strategy": "ClassWeight"},
        "RandomForest": {"source": "Best_RandomForest_SMOTE.pkl", "strategy": "SMOTE"},
        "XGBoost": {"source": "Best_XGBoost_Baseline.pkl", "strategy": "Baseline"}
    }

    for name, config in tuning_configs.items():
        source_path = os.path.join(MODEL_DIR, config['source'])
        if not os.path.exists(source_path): continue

        print(f"\n🧠 {name} 베이지안 최적화 시작...")
        
        # 전략별 데이터셋 구성
        current_X, current_y = X_train, y_train
        sample_weights = None

        if config['strategy'] == "SMOTE":
            smote = SMOTE(random_state=42, k_neighbors=3)
            current_X, current_y = smote.fit_resample(X_train, y_train)
        elif config['strategy'] == "ClassWeight" and name == "XGBoost":
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Optuna 목적 함수 정의
        def objective(trial):
            # 모델 로드
            model = joblib.load(source_path)
            
            # 파라미터 제안 (Search Space)
            if name == "DecisionTree":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                }
            elif name == "RandomForest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif name == "XGBoost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.5)
                }
            
            model.set_params(**params)
            if config['strategy'] == "ClassWeight" and name != "XGBoost":
                model.set_params(class_weight='balanced')

            # 교차 검증 점수 계산
            score = cross_val_score(model, current_X, current_y, cv=3, scoring='f1_macro', n_jobs=-1).mean()
            return score

        # 최적화 실행
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20) # 20회 시도

        print(f"✨ {name} 최적 파라미터: {study.best_params}")
        
        # 최종 모델 저장
        best_model = joblib.load(source_path)
        best_model.set_params(**study.best_params)
        if config['strategy'] == "ClassWeight" and name != "XGBoost":
            best_model.set_params(class_weight='balanced')
            
        best_model.fit(current_X, current_y, sample_weight=sample_weights if name=="XGBoost" else None)
        
        save_path = os.path.join(MODEL_DIR, f"Bayesian_Tuned_{name}.pkl")
        joblib.dump(best_model, save_path)
        
        final_f1 = f1_score(y_test, best_model.predict(X_test), average='macro')
        print(f"✅ {name} 튜닝 완료 (Macro F1: {final_f1:.4f})")

    print("\n🏁 모든 베이지안 최적화 모델 저장 완료!")

if __name__ == "__main__":
    run_bayesian_tuning_and_save()