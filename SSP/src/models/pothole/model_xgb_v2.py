"""
train_xgb_v2.py
===============
XGBoost 학습 스크립트 — 수정판
 
수정사항:
  1. scale_pos_weight 적용 (클래스 불균형 처리)
  2. eval_metric: logloss → aucpr (불균형 데이터에 적합)
  3. LOTO (Leave-One-Trip-Out) CV 구조로 평가
  4. 피처 중요도 분석 및 Top-K 재학습
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
 
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
import xgboost as xgb
 
# ── 경로 설정 ────────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
 
PROCESSED_PATH = os.path.join(project_root, "data", "processed", "pothole")
REPORT_ROOT    = os.path.join(project_root, "reports", "figures", "outdoor_performance")
 
THRESHOLD  = 0.3
DATASET_NAME = "v2"
# ─────────────────────────────────────────────────────────────────────────────
 
 
def make_save_dir(tag: str) -> str:
    now = datetime.now().strftime("%m%d_%H%M")
    path = os.path.join(REPORT_ROOT, f"xgb_{DATASET_NAME}_{tag}_{now}")
    os.makedirs(path, exist_ok=True)
    return path
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════
 
def visualize(model, X_test, y_test, save_dir: str, threshold: float = THRESHOLD, tag: str = ""):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
 
    print(f"\n{'─'*50}")
    print(f"  {tag}  |  threshold={threshold}")
    print(f"{'─'*50}")
    print(classification_report(y_test, pred, digits=3))
 
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({tag})")
    plt.savefig(os.path.join(save_dir, f"cm_{tag}.png"), dpi=150)
    plt.close()
 
    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({tag})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"roc_{tag}.png"), dpi=150)
    plt.close()
 
    # PR
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve ({tag})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"pr_{tag}.png"), dpi=150)
    plt.close()
 
    return {"auc": auc, "ap": ap}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# XGBoost 빌더 — scale_pos_weight 자동 계산
# ══════════════════════════════════════════════════════════════════════════════
 
def build_xgb(y_train: np.ndarray) -> xgb.XGBClassifier:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / (pos + 1e-9)
    print(f"  클래스 분포 — neg={neg}, pos={pos}, scale_pos_weight={spw:.2f}")
 
    return xgb.XGBClassifier(
        n_estimators    = 300,
        max_depth       = 5,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        scale_pos_weight= spw,       # ✅ 클래스 불균형 보정
        eval_metric     = "aucpr",   # ✅ 불균형에 민감한 지표
        random_state    = 42,
        n_jobs          = -1,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LOTO (Leave-One-Trip-Out) CV
# ══════════════════════════════════════════════════════════════════════════════
 
def run_loto_cv(feature_df: pd.DataFrame):
    """
    feature_df 컬럼: 피처들 + 'label' + 'trip_id'
    trip_id를 하나씩 test로 빼고 나머지로 학습.
    """
    assert "trip_id" in feature_df.columns, "feature_df에 trip_id 컬럼이 필요합니다."
    assert "label"   in feature_df.columns
 
    trips   = sorted(feature_df["trip_id"].unique())
    feat_cols = [c for c in feature_df.columns if c not in ("label", "trip_id")]
 
    all_y_true, all_y_pred, all_y_proba = [], [], []
    gain_series_list = []
    save_dir = make_save_dir("loto")
 
    for tid in trips:
        test_mask  = feature_df["trip_id"] == tid
        train_mask = ~test_mask
 
        X_train = feature_df.loc[train_mask, feat_cols]
        y_train = feature_df.loc[train_mask, "label"].values
        X_test  = feature_df.loc[test_mask,  feat_cols]
        y_test  = feature_df.loc[test_mask,  "label"].values
 
        print(f"\n▶ LOTO fold: test_trip={tid}")
        model = build_xgb(y_train)
        model.fit(X_train, y_train, verbose=False)
 
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= THRESHOLD).astype(int)
 
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(pred.tolist())
        all_y_proba.extend(proba.tolist())
 
        print(classification_report(y_test, pred, digits=3))
 
        # 피처 중요도 수집
        booster = model.get_booster()
        gain = pd.Series(booster.get_score(importance_type="gain"), name=f"trip{tid}")
        gain_series_list.append(gain)
 
    # ── 전체 집계 결과 ────────────────────────────────────────────────────────
    all_y_true  = np.array(all_y_true)
    all_y_pred  = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
 
    print("\n" + "═"*50)
    print("  LOTO 전체 집계")
    print("═"*50)
    print(classification_report(all_y_true, all_y_pred, digits=3))
    auc = roc_auc_score(all_y_true, all_y_proba)
    ap  = average_precision_score(all_y_true, all_y_proba)
    print(f"  ROC-AUC: {auc:.3f}  |  PR-AUC (AP): {ap:.3f}")
 
    # ── 피처 중요도 평균 ─────────────────────────────────────────────────────
    gain_df = pd.concat(gain_series_list, axis=1).fillna(0)
    mean_gain = gain_df.mean(axis=1).sort_values(ascending=False)
    mean_gain_pct = 100 * mean_gain / mean_gain.sum()
 
    print("\n  Top 15 Feature Importance (LOTO 평균 gain)")
    print(mean_gain_pct.head(15).to_string())
 
    plt.figure(figsize=(10, 8))
    top15 = mean_gain_pct.head(15)
    plt.barh(top15.index[::-1], top15.values[::-1])
    plt.xlabel("Mean Gain Importance (%)")
    plt.title("Feature Importance — LOTO 평균")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance_loto.png"), dpi=150)
    plt.close()
 
    return mean_gain_pct, save_dir
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Top-K 재학습 (단일 train/test split)
# ══════════════════════════════════════════════════════════════════════════════
 
def run_topk(feature_df: pd.DataFrame, mean_gain_pct: pd.Series, top_k_list=(10, 20, 30)):
    """LOTO 평균 중요도 기준 Top-K 피처로 재학습 (trip5 = test 고정)"""
    feat_cols = [c for c in feature_df.columns if c not in ("label", "trip_id")]
 
    # trip5를 test로 고정
    last_trip = sorted(feature_df["trip_id"].unique())[-1]
    test_mask  = feature_df["trip_id"] == last_trip
    train_mask = ~test_mask
 
    y_train = feature_df.loc[train_mask, "label"].values
    y_test  = feature_df.loc[test_mask,  "label"].values
 
    for k in top_k_list:
        top_feats = mean_gain_pct.head(k).index.tolist()
        # 실제 컬럼에 있는 것만
        top_feats = [f for f in top_feats if f in feat_cols]
 
        X_train = feature_df.loc[train_mask, top_feats]
        X_test  = feature_df.loc[test_mask,  top_feats]
 
        print(f"\n▶ Top-{k} 재학습 (test_trip={last_trip})")
        save_dir = make_save_dir(f"top{k}")
        model = build_xgb(y_train)
        model.fit(X_train, y_train, verbose=False)
        visualize(model, X_test, y_test, save_dir, tag=f"top{k}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    # ── 데이터 로드: full_v2.csv (trip_id 포함 전체 데이터) ──────────────────
    full_path = os.path.join(PROCESSED_PATH, "full_v2_r5.csv")
    assert os.path.exists(full_path), (
        f"full_v2.csv 없음: {full_path}\n"
        "load_raw_outdoor_v2.py를 먼저 실행해 주세요."
    )

    feature_df = pd.read_csv(full_path)
    assert "trip_id" in feature_df.columns, "full_v2.csv에 trip_id 컬럼이 없습니다."
    assert "label"   in feature_df.columns, "full_v2.csv에 label 컬럼이 없습니다."

    print(f"전체 데이터: {len(feature_df)}  |  trip 수: {feature_df['trip_id'].nunique()}")
    print(f"클래스 분포:\n{feature_df['label'].value_counts().to_string()}\n")

    # ── LOTO CV ───────────────────────────────────────────────────────────────
    mean_gain_pct, _ = run_loto_cv(feature_df)

    # ── Top-K 재학습 ──────────────────────────────────────────────────────────
    run_topk(feature_df, mean_gain_pct, top_k_list=[10, 20, 30])
 
 
if __name__ == "__main__":
    main()