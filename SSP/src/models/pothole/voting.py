import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    classification_report
)

# =======================================================
# 0. 데이터 로드
# =======================================================
DATASET_NAME = "v3_s2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")

TRAIN_PATH = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.csv")
TEST_PATH = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"✅ 학습 데이터 로드 완료 ({DATASET_NAME})\n")

# =======================================================
# 1. Base Models 정의
# =======================================================
rf_model = RandomForestClassifier(
    max_depth=5, max_features='sqrt', min_samples_split=5, 
    n_estimators=200, class_weight='balanced', random_state=42
)

# eval_metric 경고 방지를 위해 일반적인 세팅으로 조정
xgb_model = xgb.XGBClassifier(
    max_depth=6, n_estimators=500, learning_rate=0.05, 
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11.5, 
    min_child_weight=3, gamma=1, random_state=42
)

svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=1, gamma=0.01, kernel='rbf', class_weight='balanced', probability=True, random_state=42))
])

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42))
])

# =======================================================
# 2. 앙상블 실험 세팅
# =======================================================
ensembles = {
    "ENS1 (RF+XGB)": VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft', weights=[1, 1], n_jobs=-1),
    "ENS2 (SVM+XGB)": VotingClassifier(estimators=[('svm', svm_model), ('xgb', xgb_model)], voting='soft', weights=[1, 1], n_jobs=-1),
    "ENS3 (SVM+RF+XGB) 1:1:1": VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model), ('xgb', xgb_model)], voting='soft', weights=[1, 1, 1], n_jobs=-1),
    "ENS4 (SVM+RF+XGB) 1:5:1": VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model), ('xgb', xgb_model)], voting='soft', weights=[1, 5, 1], n_jobs=-1),
    "ENS5 (LR+SVM+RF+XGB) 1:1:1:1": VotingClassifier(estimators=[('lr', lr_model), ('svm', svm_model), ('rf', rf_model), ('xgb', xgb_model)], voting='soft', weights=[1, 1, 1, 1], n_jobs=-1),

}

# =======================================================
# 3. 모델 학습 및 실시간 결과 출력
# =======================================================
print("🚀 앙상블 모델 학습 및 평가 시작 (시간이 조금 걸릴 수 있습니다)...\n")

results = []

for ens_name, clf in ensembles.items():
    print(f"⏳ 학습 중: {ens_name} ...")
    
    # 학습
    clf.fit(X_train, y_train)
    
    # 예측
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # 최적 임계값 찾기
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    
    y_pred = (y_probs >= best_thresh).astype(int)
    
    # 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # 💡 결과를 즉시 화면에 출력!
    print("="*60)
    print(f"🎯 {ens_name} 완료! (최적 임계값: {best_thresh:.4f})")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))
    print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    print("="*60 + "\n")
    
    results.append({
        "Model": ens_name.split()[0], # ENS1, ENS2 등 이름만 추출
        "Threshold": best_thresh,
        "F1-Score": f1,
        "Recall": rec,
        "Precision": prec,
        "PR-AUC": pr_auc
    })

# =======================================================
# 4. 전체 결과 비교 요약 및 시각화
# =======================================================