import os
import sys
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 경로 설정 및 데이터 로더 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)

from data.routes_indoor import load_processed_data

def train_and_save():
    model_save_path = r"C:\work\mlproject\Surface-Perception-System\SSP\models\indoor"
    os.makedirs(model_save_path, exist_ok=True)

    print("🚀 데이터를 불러오는 중...")
    X_train, _, y_train, _ = load_processed_data() # 테스트셋은 여기서 사용 안 함

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    print("\n--- 모델 학습 시작 ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        save_name = f"Base_{name.replace(' ', '_')}.pkl"
        joblib.dump(model, os.path.join(model_save_path, save_name))
        print(f"✅ {name:20} 저장 완료")

if __name__ == "__main__":
    train_and_save()