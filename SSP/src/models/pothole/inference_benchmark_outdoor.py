"""
inference_benchmark_outdoor.py
==============================
실외(Pothole) 모델 추론 속도 및 모델 크기 비교 벤치마크 스크립트

비교 대상:
  - 단일 모델 4종: XGBoost, Random Forest, SVM, Logistic Regression
  - 앙상블 2종: Voting (Soft), Stacking

측정 항목:
  1. 추론 속도 (1000회 반복 평균, ms)
  2. 모델 파일 크기 (MB)
  3. 핵심 성능 지표 (F1, Recall, PR-AUC)

출력:
  - 콘솔 요약 테이블
  - 바 차트 (추론 속도 비교)
  - 버블 차트 (속도 vs 크기 vs 성능)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score, recall_score, average_precision_score, precision_recall_curve

# =======================================================
# 0. 경로 설정
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/outdoor")
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/final_comparison")
os.makedirs(FIGURE_DIR, exist_ok=True)

DATASET_NAME = "v3_s2"
N_REPEATS = 1000  # 추론 반복 횟수

# =======================================================
# 1. 테스트 데이터 로드
# =======================================================
print("📂 테스트 데이터 로드 중...")
test_df = pd.read_csv(os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv"))
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
print(f"✅ 테스트 데이터 로드 완료 (샘플 수: {len(X_test)}, 피처 수: {X_test.shape[1]})\n")

# =======================================================
# 2. 모델 정의 (파일명, 설명, 스케일러, 임계값)
# =======================================================
# 사용자님이 보신 이미지(03_final_Metrics_Bar.png)와 수치를 일치시키기 위해 
# graph.py에 정의된 고정 임계값을 사용합니다.
model_configs = {
    "XGBoost (Tuned)": {
        "model_file": "XGB1_best_model.pkl",
        "scaler_file": None,
        "is_dict": True,
        "threshold": 0.5
    },
    "Random Forest (Class+Grid)": {
        "model_file": "rf_class_grid_v3_s2.pkl",
        "scaler_file": None,
        "is_dict": False,
        "threshold": 0.4528
    },
    "SVM (ClassWeight+Grid)": {
        "model_file": "svm_class_grid_v3_s2.pkl",
        "scaler_file": "svm_scaler_v3_s2.pkl",
        "is_dict": False,
        "threshold": 0.2083
    },
    "Logistic (ClassWeight+Grid)": {
        "model_file": "logistic_class_grid_v3_s2.pkl",
        "scaler_file": "logistic_scaler_v3_s2.pkl",
        "is_dict": False,
        "threshold": 0.5152
    },
    "Voting (Soft Ensemble)": {
        "model_file": "voting_soft_v3_s2.pkl",
        "scaler_file": None,
        "is_dict": False,
        "threshold": 0.5
    },
    "Stacking (Meta-Learner)": {
        "model_file": "stacking_best_v3_s2.pkl",
        "scaler_file": None,
        "is_dict": False,
        "threshold": 0.5
    },
}

# =======================================================
# 3. 모델 로드 및 벤치마크 실행
# =======================================================
print("🚀 추론 속도 벤치마크 시작 (기존 임계값 적용)...\n")
print("=" * 85)
print(f"{'Model Name':<30} | {'Speed (ms)':<15} | {'Size':<10} | {'Recall':<6} | {'F1':<6} | {'Thresh':<6}")
print("-" * 85)

results = []

for name, config in model_configs.items():
    model_path = os.path.join(MODEL_DIR, config["model_file"])

    if not os.path.exists(model_path):
        continue

    # --- 모델 로드 ---
    raw = joblib.load(model_path)
    model = raw["model"] if config["is_dict"] else raw

    # --- 스케일러 로드 ---
    scaler = None
    if config["scaler_file"]:
        scaler_path = os.path.join(MODEL_DIR, config["scaler_file"])
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

    # --- 모델 크기 계산 ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if config["scaler_file"]:
        scaler_path = os.path.join(MODEL_DIR, config["scaler_file"])
        if os.path.exists(scaler_path):
            model_size_mb += os.path.getsize(scaler_path) / (1024 * 1024)

    # --- 데이터 준비 ---
    X_input = scaler.transform(X_test) if scaler else (X_test.values if hasattr(X_test, 'values') else X_test)

    # --- 고정 임계값 적용 ---
    threshold = config["threshold"]
    y_probs = model.predict_proba(X_input)[:, 1]

    # --- 추론 속도 측정 (1000회 반복) ---
    times = []
    # 워밍업
    model.predict_proba(X_input)
    
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        if scaler:
            X_s = scaler.transform(X_test)
            model.predict_proba(X_s)
        else:
            model.predict_proba(X_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    # --- 최종 지표 계산 ---
    y_pred = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_probs)

    results.append({
        "Model": name,
        "Avg_ms": avg_ms,
        "Std_ms": std_ms,
        "Size_MB": model_size_mb,
        "F1": f1,
        "Recall": recall,
        "PR_AUC": pr_auc,
        "Threshold": threshold,
    })

    print(f"✅ {name:<27s} | {avg_ms:6.2f}±{std_ms:4.2f} ms | {model_size_mb:6.2f} MB | {recall:6.3f} | {f1:6.3f} | {threshold:6.3f}")

print("-" * 85)

# =======================================================
# 4. 결과 DataFrame
# =======================================================
df = pd.DataFrame(results)
print(f"\n📊 벤치마크 결과 요약 ({N_REPEATS}회 반복 평균):\n")
print(df.to_string(index=False))

# CSV 저장
csv_path = os.path.join(FIGURE_DIR, "outdoor_inference_benchmark.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n💾 CSV 저장 완료: {csv_path}")

# =======================================================
# 5. 시각화 1 — 추론 속도 비교 바 차트
# =======================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))

# 모델 타입별 색상 구분 (단일 vs 앙상블)
colors = []
for m in df['Model']:
    if 'Voting' in m or 'Stacking' in m:
        colors.append('#e74c3c')  # 앙상블: 빨간 계열
    else:
        colors.append('#3498db')  # 단일 모델: 파란 계열

bars = ax.bar(df['Model'], df['Avg_ms'], yerr=df['Std_ms'],
              color=colors, edgecolor='white', linewidth=1.5,
              capsize=5, error_kw={'linewidth': 1.5})

# 막대 위에 수치 표시
for bar, val, std in zip(bars, df['Avg_ms'], df['Std_ms']):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.3,
            f'{val:.2f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Outdoor Model Inference Speed Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Inference Time (ms)', fontsize=12)
ax.set_xlabel('')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 범례 (단일 vs 앙상블)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='white', label='Single Model'),
    Patch(facecolor='#e74c3c', edgecolor='white', label='Ensemble Model')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True)

plt.tight_layout()
save_path1 = os.path.join(FIGURE_DIR, "05_outdoor_inference_speed.png")
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
print(f"\n📈 추론 속도 비교 그래프 저장: {save_path1}")
plt.close()

# =======================================================
# 6. 시각화 2 — 속도 vs 크기 vs 성능 버블 차트
# =======================================================
fig, ax = plt.subplots(figsize=(12, 7))

# 버블 크기: F1-Score 기반 (0~1 → 300~1500)
bubble_sizes = (df['F1'] * 1200) + 300

# 색상: Recall 기반
scatter = ax.scatter(
    df['Avg_ms'], df['Size_MB'],
    s=bubble_sizes, c=df['Recall'], cmap='RdYlGn',
    alpha=0.8, edgecolors='black', linewidths=1.5,
    vmin=0.4, vmax=1.0
)

# 모델명 라벨
for i, row in df.iterrows():
    ax.annotate(
        f"{row['Model']}\n(F1:{row['F1']:.3f})",
        (row['Avg_ms'], row['Size_MB']),
        textcoords="offset points", xytext=(12, 8),
        fontsize=10, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
    )

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Recall (Pothole Detection Rate)', fontsize=11)

ax.set_title('Outdoor Model: Speed vs Size vs Performance', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Inference Time (ms) — 낮을수록 빠름', fontsize=12)
ax.set_ylabel('Model Size (MB) — 낮을수록 가벼움', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

# 이상적 영역 표시 (좌하단이 좋음)
ax.annotate('⭐ Ideal Zone\n(Fast & Light)',
            xy=(0.05, 0.05), xycoords='axes fraction',
            fontsize=12, color='green', fontweight='bold', alpha=0.6)

plt.tight_layout()
save_path2 = os.path.join(FIGURE_DIR, "06_outdoor_speed_vs_size.png")
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"📈 속도 vs 크기 버블 차트 저장: {save_path2}")
plt.close()

print(f"\n🎉 실외 모델 벤치마크 완료!")
