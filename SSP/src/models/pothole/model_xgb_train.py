"""
train_xgb_v3.py
===============
XGBoost 성능 개선 실험 스크립트 — XGB1 ~ XGB8 전체 실험 조합

수정사항 (v3 → v3 fix):
  1. BASE 파라미터를 D1~D5 기준과 동일하게 고정
     (max_depth=6, n_estimators=500, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8,
      scale_pos_weight=11.5, min_child_weight=3, gamma=1,
      eval_metric='logloss')
  2. XGB1 baseline도 scale_pos_weight=11.5 포함 (D5와 동일 조건)
  3. Threshold 탐색: train 확률 → OOF(Out-Of-Fold) 확률 기반으로 수정
  4. RandomizedSearchCV scoring: average_precision → f1
  5. CV: StratifiedKFold → RepeatedStratifiedKFold (소수 클래스 안정화)

실험 조합:
  XGB1 : Baseline — D1~D5와 완전히 동일한 파라미터
  XGB2 : + scale_pos_weight 자동 계산 비교
  XGB3 : + GridSearchCV (D5 파라미터를 시작점으로 탐색)
  XGB4 : + GridSearchCV + Threshold Tuning (OOF 기반)
  XGB5 : + GridSearchCV + Threshold + Top-10 Feature
  XGB6 : + GridSearchCV + Threshold + Top-20 Feature
  XGB7 : + GridSearchCV + Threshold + Top-30 Feature
  XGB8 : + GridSearchCV + Threshold + Top-50 Feature
"""

import os
import warnings
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
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── 경로 설정 ────────────────────────────────────────────────────────────────
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

PROCESSED_PATH = os.path.join(project_root, "data", "processed", "pothole")
REPORT_ROOT    = os.path.join(project_root, "reports", "figures", "outdoor_performance")

DATASET_NAME = "v3_s2"
RUN_TS       = datetime.now().strftime("%m%d_%H%M")

# ── D1~D5에서 사용한 기준 파라미터 (고정) ────────────────────────────────────
BASE_PARAMS = dict(
    max_depth        = 6,
    n_estimators     = 500,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = 11.5,   # D1~D5 고정값
    min_child_weight = 3,
    gamma            = 1,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1,
)

# ── GridSearch 탐색 공간 (BASE_PARAMS 주변 탐색) ──────────────────────────────
PARAM_GRID = {
    "max_depth"       : [4, 5, 6, 7],
    "n_estimators"    : [300, 500, 700],
    "learning_rate"   : [0.03, 0.05, 0.1],
    "subsample"       : [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma"           : [0, 0.5, 1, 2],
}

# ✅ RepeatedStratifiedKFold: 5fold × 3repeat (소수 클래스 안정화)
CV_SPLITS      = 5
CV_REPEATS     = 3
RANDOM_ITER    = 50
RANDOM_STATE   = 42
DEFAULT_THRESH = 0.5
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_save_dir(exp_id: str) -> str:
    path = os.path.join(REPORT_ROOT, f"{exp_id}_{DATASET_NAME}_{RUN_TS}")
    os.makedirs(path, exist_ok=True)
    return path


def build_base_xgb(scale_pos_weight: float = None,
                   extra_params: dict = None) -> xgb.XGBClassifier:
    """
    D1~D5 기준 파라미터로 XGBoost 생성.
    scale_pos_weight를 명시하면 BASE_PARAMS의 11.5를 덮어씀.
    """
    params = BASE_PARAMS.copy()
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    if extra_params:
        params.update(extra_params)
    return xgb.XGBClassifier(**params)


def compute_spw(y: np.ndarray) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    spw = neg / (pos + 1e-9)
    print(f"  클래스 분포 — neg={neg}, pos={pos}, 자동 scale_pos_weight={spw:.2f}")
    return spw


# ══════════════════════════════════════════════════════════════════════════════
# Threshold 최적화 — OOF 확률 기반 (train 과적합 방지)
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold_oof(model, X_train, y_train) -> float:
    """
    ✅ OOF(Out-Of-Fold) 확률로 threshold 탐색.
    train 확률로 탐색하면 threshold가 0.9↑로 과적합됨.
    """
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        model, X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    prec, rec, thresholds = precision_recall_curve(y_train, oof_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    print(f"  ✅ OOF 기반 최적 threshold={best_thr:.3f}  "
          f"(Prec={prec[best_idx]:.3f}, Rec={rec[best_idx]:.3f}, F1={f1s[best_idx]:.3f})")
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 & 지표 계산
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, save_dir: str,
             threshold: float = DEFAULT_THRESH, tag: str = "") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)

    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec  = recall_score(y_test, pred, zero_division=0)
    f1   = f1_score(y_test, pred, zero_division=0)
    auc  = roc_auc_score(y_test, proba)
    ap   = average_precision_score(y_test, proba)

    print(f"\n{'─'*55}")
    print(f"  {tag}  |  threshold={threshold:.3f}")
    print(f"{'─'*55}")
    print(classification_report(y_test, pred, digits=3))
    print(f"  ROC-AUC={auc:.3f}  |  PR-AUC={ap:.3f}")

    slug = tag.replace(" ", "_").replace("+", "")

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{slug}.png"), dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_{slug}.png"), dpi=150)
    plt.close()

    # PR Curve
    p_, r_, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(r_, p_, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_{slug}.png"), dpi=150)
    plt.close()

    # Threshold 스캔
    _plot_threshold_scan(y_test, proba, save_dir, slug, threshold)

    return dict(tag=tag, threshold=threshold, accuracy=acc,
                precision=prec, recall=rec, f1=f1, roc_auc=auc, pr_auc=ap)


def _plot_threshold_scan(y_true, y_proba, save_dir, slug, best_thr):
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        p = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, p, zero_division=0))
        precs.append(precision_score(y_true, p, zero_division=0))
        recs.append(recall_score(y_true, p, zero_division=0))
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, f1s,   label="F1")
    plt.plot(thresholds, precs, label="Precision", linestyle="--")
    plt.plot(thresholds, recs,  label="Recall",    linestyle=":")
    plt.axvline(best_thr, color="red", linestyle=":", linewidth=1,
                label=f"used thr={best_thr:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title(f"Threshold Scan ({slug})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"thr_scan_{slug}.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feat_cols, save_dir, slug, top_k=30):
    booster = model.get_booster()
    gain = pd.Series(booster.get_score(importance_type="gain"), name="gain")
    gain = gain.reindex(feat_cols).fillna(0)
    gain_pct = (100 * gain / gain.sum()).sort_values(ascending=False)
    top = gain_pct.head(top_k)
    plt.figure(figsize=(10, max(4, top_k * 0.35)))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.xlabel("Gain Importance (%)")
    plt.title(f"Feature Importance — {slug}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fi_{slug}.png"), dpi=150)
    plt.close()
    return gain_pct


# ══════════════════════════════════════════════════════════════════════════════
# GridSearch — ✅ scoring=f1, ✅ RepeatedStratifiedKFold
# ══════════════════════════════════════════════════════════════════════════════

def run_randomized_search(X_train, y_train,
                          scale_pos_weight: float = 11.5) -> tuple:
    base = build_base_xgb(scale_pos_weight=scale_pos_weight)

    # ✅ RepeatedStratifiedKFold: fold당 pos 샘플 부족 문제 완화
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator          = base,
        param_distributions= PARAM_GRID,
        n_iter             = RANDOM_ITER,
        scoring            = "f1",        # ✅ 실제 목표 지표와 일치
        cv                 = cv,
        n_jobs             = -1,
        random_state       = RANDOM_STATE,
        verbose            = 1,
        refit              = True,
    )
    print(f"  RandomizedSearchCV 시작 "
          f"(n_iter={RANDOM_ITER}, {CV_SPLITS}fold×{CV_REPEATS}repeat, scoring=f1)…")
    search.fit(X_train, y_train)
    print(f"  Best F1(CV)={search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ══════════════════════════════════════════════════════════════════════════════
# LOTO 피처 중요도 — BASE_PARAMS 사용
# ══════════════════════════════════════════════════════════════════════════════

def run_loto_importance(feature_df: pd.DataFrame, feat_cols: list) -> pd.Series:
    """LOTO fold별 gain 중요도 평균. BASE_PARAMS(D5 기준) 고정 사용."""
    trips = sorted(feature_df["trip_id"].unique())
    gain_list = []
    for tid in trips:
        mask_tr = feature_df["trip_id"] != tid
        X_tr = feature_df.loc[mask_tr, feat_cols]
        y_tr = feature_df.loc[mask_tr, "label"].values
        model = build_base_xgb()          # scale_pos_weight=11.5 고정
        model.fit(X_tr, y_tr, verbose=False)
        gain = pd.Series(
            model.get_booster().get_score(importance_type="gain"),
            name=f"trip{tid}"
        )
        gain_list.append(gain)
    gain_df   = pd.concat(gain_list, axis=1).fillna(0)
    mean_gain = gain_df.mean(axis=1).sort_values(ascending=False)
    return 100 * mean_gain / mean_gain.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 실험 러너
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_id: str,
                   X_train, y_train, X_test, y_test,
                   feat_cols: list,
                   use_auto_spw: bool   = False,
                   use_grid: bool       = False,
                   use_threshold: bool  = False,
                   top_k: int           = None,
                   mean_gain_pct: pd.Series = None) -> dict:

    save_dir  = make_save_dir(exp_id)
    tag_parts = [exp_id]

    # ── Top-K 피처 선택 ───────────────────────────────────────────────────────
    if top_k is not None and mean_gain_pct is not None:
        selected  = [f for f in mean_gain_pct.head(top_k).index if f in feat_cols]
        print(f"\n  Top-{top_k} 피처 선택: {len(selected)}개")
        X_train   = X_train[selected]
        X_test    = X_test[selected]
        feat_cols = selected
        tag_parts.append(f"Top{top_k}")

    # ── scale_pos_weight ──────────────────────────────────────────────────────
    if use_auto_spw:
        spw = compute_spw(y_train)
        tag_parts.append("AutoSPW")
    else:
        spw = 11.5
        print(f"  scale_pos_weight=11.5 (D5 기준 고정)")

    # ── 학습 ──────────────────────────────────────────────────────────────────
    best_params = {}
    if use_grid:
        model, best_params = run_randomized_search(X_train, y_train,
                                                   scale_pos_weight=spw)
        tag_parts.append("Grid")
    else:
        model = build_base_xgb(scale_pos_weight=spw)
        model.fit(X_train, y_train, verbose=False)

    # ── Threshold 결정 — ✅ OOF 기반 ─────────────────────────────────────────
    if use_threshold:
        threshold = find_best_threshold_oof(model, X_train, y_train)
        tag_parts.append("Thr")
    else:
        threshold = DEFAULT_THRESH

    # ── 피처 중요도 ───────────────────────────────────────────────────────────
    fi = plot_feature_importance(model, feat_cols, save_dir, exp_id)
    top_feats_str = ", ".join(fi.head(5).index.tolist())

    # ── 평가 ──────────────────────────────────────────────────────────────────
    tag_label = " + ".join(tag_parts)
    metrics   = evaluate(model, X_test, y_test, save_dir,
                         threshold=threshold, tag=tag_label)
    metrics.update(dict(
        exp_id       = exp_id,
        use_auto_spw = use_auto_spw,
        spw_value    = spw,
        use_grid     = use_grid,
        use_threshold= use_threshold,
        top_k        = top_k if top_k else "전체",
        best_params  = str(best_params) if best_params else "-",
        top_features = top_feats_str,
    ))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results: list, save_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    col_order = [
        "exp_id", "spw_value", "use_auto_spw", "use_grid", "use_threshold",
        "top_k", "threshold", "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "best_params", "top_features"
    ]
    df = df[[c for c in col_order if c in df.columns]]
    path = os.path.join(save_dir, "experiment_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n{'═'*65}")
    print("  실험 결과 요약")
    print("═"*65)
    print(df[["exp_id", "spw_value", "threshold", "accuracy",
              "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
    print(f"\n  ✅ 결과 저장: {path}")

    for metric, color, title in [
        ("f1",     "steelblue",  "F1 Score"),
        ("pr_auc", "darkorange", "PR-AUC"),
        ("recall", "seagreen",   "Recall"),
    ]:
        plt.figure(figsize=(10, 4))
        plt.bar(df["exp_id"], df[metric], color=color)
        plt.axhline(df[metric].max(), color="red", linestyle="--", linewidth=0.8,
                    label=f"Best={df[metric].max():.3f}")
        plt.xlabel("Experiment"); plt.ylabel(title)
        plt.title(f"{title} Comparison — XGB1~XGB8")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"summary_{metric}.png"), dpi=150)
        plt.close()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    train_df = pd.read_csv(os.path.join(PROCESSED_PATH, "train_v3_s2.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_PATH, "test_v3_s2.csv"))
    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

    feat_cols = [c for c in train_df.columns if c not in ("label", "trip_id")]
    X_train   = train_df[feat_cols]
    y_train   = train_df["label"].values
    X_test    = test_df[feat_cols]
    y_test    = test_df["label"].values

    print(f"\n피처 수: {len(feat_cols)}")
    print(f"클래스 분포(Train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"클래스 분포(Test) : {dict(zip(*np.unique(y_test,  return_counts=True)))}")
    print(f"\n기준 파라미터(D5 동일):\n{BASE_PARAMS}")

    summary_dir = make_save_dir("SUMMARY")

    # ── 피처 중요도 사전 계산 ─────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  피처 중요도 사전 계산 (BASE_PARAMS 기준)")
    print("═"*55)

    if "trip_id" in train_df.columns:
        feature_df    = pd.concat([train_df, test_df], ignore_index=True)
        mean_gain_pct = run_loto_importance(feature_df, feat_cols)
    else:
        tmp = build_base_xgb()
        tmp.fit(X_train, y_train, verbose=False)
        gain = pd.Series(
            tmp.get_booster().get_score(importance_type="gain")
        ).sort_values(ascending=False)
        mean_gain_pct = 100 * gain / gain.sum()

    print("\n  Top 20 Feature Importance:")
    print(mean_gain_pct.head(20).to_string())

    results = []

    # XGB1 — D1~D5 완전 동일 파라미터 (비교 기준선)
    print("\n" + "═"*55)
    print("  XGB1 — Baseline (D5와 동일 파라미터)")
    print("═"*55)
    results.append(run_experiment(
        "XGB1", X_train.copy(), y_train, X_test.copy(), y_test, feat_cols,
        use_auto_spw=False, use_grid=False, use_threshold=False))

    # XGB2 — scale_pos_weight 자동 계산 비교
    print("\n" + "═"*55)
    print("  XGB2 — scale_pos_weight 자동 계산 비교")
    print("═"*55)
    results.append(run_experiment(
        "XGB2", X_train.copy(), y_train, X_test.copy(), y_test, feat_cols,
        use_auto_spw=True, use_grid=False, use_threshold=False))

    # XGB3 — + GridSearchCV
    print("\n" + "═"*55)
    print("  XGB3 — + GridSearchCV (scoring=f1, RepeatedStratifiedKFold)")
    print("═"*55)
    results.append(run_experiment(
        "XGB3", X_train.copy(), y_train, X_test.copy(), y_test, feat_cols,
        use_auto_spw=False, use_grid=True, use_threshold=False))

    # XGB4 — + GridSearchCV + Threshold Tuning (OOF 기반)
    print("\n" + "═"*55)
    print("  XGB4 — + GridSearchCV + Threshold Tuning (OOF 기반)")
    print("═"*55)
    results.append(run_experiment(
        "XGB4", X_train.copy(), y_train, X_test.copy(), y_test, feat_cols,
        use_auto_spw=False, use_grid=True, use_threshold=True))

    # XGB5~8 — Top-K Feature
    for exp_id, top_k in [("XGB5", 10), ("XGB6", 20), ("XGB7", 30), ("XGB8", 50)]:
        print("\n" + "═"*55)
        print(f"  {exp_id} — + Grid + Threshold (OOF) + Top-{top_k}")
        print("═"*55)
        results.append(run_experiment(
            exp_id, X_train.copy(), y_train, X_test.copy(), y_test, feat_cols,
            use_auto_spw=False, use_grid=True, use_threshold=True,
            top_k=top_k, mean_gain_pct=mean_gain_pct))

    # 전체 요약
    summary_df = save_summary(results, summary_dir)
    best = summary_df.loc[summary_df["f1"].idxmax()]
    print(f"\n  🏆 Best: {best['exp_id']}  "
          f"F1={best['f1']:.3f}  Recall={best['recall']:.3f}  "
          f"Precision={best['precision']:.3f}  PR-AUC={best['pr_auc']:.3f}")

    # ✅ Best 모델 다시 학습 후 저장
    import joblib
    print("\n  🔄 Best 모델 재학습 및 저장...")

    # feature 다시 준비
    train_df = pd.read_csv(os.path.join(PROCESSED_PATH, "train_v3_s2.csv"))
    feat_cols = [c for c in train_df.columns if c not in ("label", "trip_id")]
    X_train = train_df[feat_cols]
    y_train = train_df["label"].values

    # 모델 생성 (XGB1 기준 → 가장 성능 좋았음)
    best_model = build_base_xgb(scale_pos_weight=11.5)
    best_model.fit(X_train, y_train)

    # 저장 경로
    model_path = os.path.join(project_root, "models", "pothole")
    os.makedirs(model_path, exist_ok=True)

    save_path = os.path.join(model_path, f"{best['exp_id']}_best_model.pkl")

    best_threshold = best["threshold"]

    joblib.dump({
        "model": best_model,
        "threshold": best_threshold
    }, save_path)
    print(f"  ✅ 모델, 임계값 저장 완료 → {save_path}")

    # 데이터 로드 시
    # data = joblib.load(save_path)
    # model = data["model"]
    # threshold = data["threshold"]
    
if __name__ == "__main__":
    main()