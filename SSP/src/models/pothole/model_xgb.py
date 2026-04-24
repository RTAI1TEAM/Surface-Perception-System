# XGBoost Full Pipeline (Auto Save Version)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    accuracy_score
)
import xgboost as xgb
import joblib


# ======================
# 저장 폴더 자동 생성
# ======================
def make_save_dir(project_root, dataset_name, model_name, threshold, top_k=None):
    now = datetime.now().strftime("%m%d_%H%M")

    if top_k:
        folder_name = f"{model_name}_{dataset_name}_top{top_k}_thr{threshold}_{now}"
    else:
        folder_name = f"{model_name}_{dataset_name}_thr{threshold}_{now}"

    save_dir = os.path.join(
        project_root,
        "reports",
        "figures",
        "outdoor_performance",
        folder_name
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ======================
# 시각화 함수
# ======================
def visualization(model, x_test, y_test, save_dir, threshold=0.3):
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba > threshold).astype(int)

    print(f"\n===== threshold = {threshold} =====")

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Report
    print(classification_report(y_test, pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=300)
    plt.close()

    print('acc', accuracy_score(y_test, pred))

# ======================
# 데이터 로드
# ======================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

BASE_PATH = os.path.join(project_root, "data", "processed", "pothole")

TRAIN_PATH = os.path.join(BASE_PATH, "train_v2.csv")
TEST_PATH = os.path.join(BASE_PATH, "test_v2.csv")

DATASET_NAME = "v2"
MODEL_NAME = "xgb"
THRESHOLD = 0.3

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

x_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

x_test = test_data.drop("label", axis=1)
y_test = test_data["label"]


# ======================
# 모델 학습 (전체 feature)
# ======================
SAVE_DIR = make_save_dir(project_root, DATASET_NAME, MODEL_NAME, THRESHOLD)

xgb_clf = xgb.XGBClassifier(
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=11.5,
    min_child_weight=3,
    gamma=1,
    eval_metric='logloss',
    random_state=42
)


xgb_clf.fit(x_train, y_train)

print("\n===== XGBoost (All Features) =====")
visualization(xgb_clf, x_test, y_test, SAVE_DIR, THRESHOLD)

# ======================
# 모델 저장 (joblib)
# ======================
model_path = os.path.join(project_root, "models", "pothole", "xgb_tuning_model.pkl")
joblib.dump(xgb_clf, model_path)

print(f"모델 저장 완료 → {model_path}")


# ======================
# Feature Importance
# ======================
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain'))

xgb_gain_pct = 100 * xgb_gain / (xgb_gain.sum() if xgb_gain.sum() != 0 else 1)
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)
xgb_gain_pct = xgb_gain_pct.sort_values(ascending=False)

print("\n===== Top Feature Importance =====")
print(xgb_gain_pct.head(10))

# 전체 중요도 시각화
plt.figure(figsize=(10, 12))
plt.barh(xgb_gain_pct.index[::-1], xgb_gain_pct.values[::-1])
plt.title("Feature Importance (All)")
plt.xlabel("Importance (%)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_importance_all.png"), dpi=300)
plt.close()


# ======================
# Top-K Feature 선택 후 재학습
# ======================
# for k in [10, 20, 30]:
#     TOP_K = k

#     top_features = xgb_gain_pct.head(TOP_K).index.tolist()

#     print("\n===== Top Features =====")
#     print(top_features)

#     x_train_top = x_train[top_features]
#     x_test_top = x_test[top_features]

#     SAVE_DIR_TOP = make_save_dir(project_root, DATASET_NAME, MODEL_NAME, THRESHOLD, TOP_K)

#     xgb_top = xgb.XGBClassifier(
#         max_depth=6,
#         n_estimators=200,
#         eval_metric='logloss',
#         random_state=42
#     )

#     xgb_top.fit(x_train_top, y_train)

#     print("\n===== XGBoost (Top Features) =====")
#     visualization(xgb_top, x_test_top, y_test, SAVE_DIR_TOP, THRESHOLD)