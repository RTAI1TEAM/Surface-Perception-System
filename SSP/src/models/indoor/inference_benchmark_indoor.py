"""
inference_benchmark_indoor.py
=============================
실내(Indoor) 모델 추론 속도 및 모델 크기 비교 벤치마크 스크립트

비교 대상:
  - DecisionTree (Final_Best)
  - RandomForest (Final_Best)
  - XGBoost (Final_Best)

측정 항목:
  1. 추론 속도 (1000회 반복 평균, ms)
  2. 모델 파일 크기 (MB)
  3. 핵심 성능 지표 (Macro F1, Accuracy)

출력:
  - 콘솔 요약 테이블
  - 바 차트 (추론 속도 비교)
  - 버블 차트 (속도 vs 크기 vs 성능)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score, accuracy_score, recall_score

# =======================================================
# 0. 경로 설정 및 데이터 로더 임포트
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/indoor")
DATA_PATH = os.path.join(BASE_DIR, "../../../data/processed/indoor/indoor_train_features.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/indoor_performance")
os.makedirs(FIGURE_DIR, exist_ok=True)

# routes_indoor 모듈 임포트를 위한 경로 설정
project_src = os.path.abspath(os.path.join(BASE_DIR, "../../"))
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

N_REPEATS = 1000  # 추론 반복 횟수

# =======================================================
# 1. 테스트 데이터 로드
# =======================================================
print("📂 테스트 데이터 로드 중...")
_, X_test, _, y_test = load_processed_data(DATA_PATH)
print(f"✅ 테스트 데이터 로드 완료 (샘플 수: {len(X_test)}, 피처 수: {X_test.shape[1]})")
print(f"   클래스 분포: {dict(y_test.value_counts())}\n")

# =======================================================
# 2. 비교 대상 모델 정의 (라벨 보강)
# =======================================================
model_configs = {
    "XGBoost (Base)": "Base_XGBoost.pkl",
    "XGBoost (HyperTuned+Final)": "Final_Best_XGBoost.pkl",
    "RandomForest (SMOTE+Final)": "Final_Best_RandomForest.pkl",
    "DecisionTree (ClassWeight+Final)": "Final_Best_DecisionTree.pkl",
}

# =======================================================
# 3. 벤치마크 실행
# =======================================================
print("🚀 추론 속도 벤치마크 시작...\n")
print("=" * 70)

results = []

for name, model_file in model_configs.items():
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        print(f"⚠️  {name}: 모델 파일 없음 ({model_file}) — 건너뜁니다.")
        continue

    # --- 모델 로드 ---
    model = joblib.load(model_path)

    # --- 모델 크기 계산 ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # --- 입력 데이터 준비 ---
    X_input = X_test.values if hasattr(X_test, 'values') else X_test

    # --- 워밍업 ---
    model.predict(X_input)

    # --- 추론 속도 측정 ---
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        model.predict(X_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms 변환

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    # --- 성능 지표 계산 (다중 분류: macro 기준) ---
    y_pred = model.predict(X_input)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')

    results.append({
        "Model": name,
        "Avg_ms": avg_ms,
        "Std_ms": std_ms,
        "Size_MB": model_size_mb,
        "F1_Macro": f1,
        "Accuracy": acc,
        "Recall_Macro": recall,
    })

    # 실시간 출력
    print(f"✅ {name:15s} | {avg_ms:8.2f} ± {std_ms:.2f} ms | "
          f"Size: {model_size_mb:7.2f} MB | F1(Macro): {f1:.3f} | Acc: {acc:.3f}")

print("=" * 70)

# =======================================================
# 4. 결과 DataFrame
# =======================================================
df = pd.DataFrame(results)
print(f"\n📊 벤치마크 결과 요약 ({N_REPEATS}회 반복 평균):\n")
print(df.to_string(index=False))

# CSV 저장
csv_path = os.path.join(FIGURE_DIR, "indoor_inference_benchmark.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n💾 CSV 저장 완료: {csv_path}")

# =======================================================
# 5. 시각화 1 — 추론 속도 비교 바 차트
# =======================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# 모델별 색상
palette = {'DecisionTree': '#2ecc71', 'RandomForest': '#e67e22', 'XGBoost': '#3498db'}
colors = [palette.get(m, '#95a5a6') for m in df['Model']]

bars = ax.bar(df['Model'], df['Avg_ms'], yerr=df['Std_ms'],
              color=colors, edgecolor='white', linewidth=1.5,
              capsize=5, error_kw={'linewidth': 1.5})

# 막대 위에 수치 표시
for bar, val, std in zip(bars, df['Avg_ms'], df['Std_ms']):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.2,
            f'{val:.2f}ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Indoor Model Inference Speed Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Inference Time (ms)', fontsize=12)
ax.set_xlabel('')
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
save_path1 = os.path.join(FIGURE_DIR, "inference_speed_comparison.png")
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
print(f"\n📈 추론 속도 비교 그래프 저장: {save_path1}")
plt.close()

# =======================================================
# 6. 시각화 2 — 속도 vs 크기 vs 성능 버블 차트
# =======================================================
fig, ax = plt.subplots(figsize=(10, 7))

# 버블 크기: F1-Score 기반 (0~1 → 400~1600)
bubble_sizes = (df['F1_Macro'] * 1200) + 400

# 색상: F1 Macro 기반
scatter = ax.scatter(
    df['Avg_ms'], df['Size_MB'],
    s=bubble_sizes, c=df['F1_Macro'], cmap='RdYlGn',
    alpha=0.8, edgecolors='black', linewidths=1.5,
    vmin=0.6, vmax=1.0
)

# 모델명 라벨
for i, row in df.iterrows():
    ax.annotate(
        f"{row['Model']}\n(F1:{row['F1_Macro']:.3f})",
        (row['Avg_ms'], row['Size_MB']),
        textcoords="offset points", xytext=(15, 10),
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
    )

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Macro F1-Score', fontsize=11)

ax.set_title('Indoor Model: Speed vs Size vs Performance', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Inference Time (ms) — 낮을수록 빠름', fontsize=12)
ax.set_ylabel('Model Size (MB) — 낮을수록 가벼움', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

# 이상적 영역 표시
ax.annotate('⭐ Ideal Zone\n(Fast & Light)',
            xy=(0.05, 0.05), xycoords='axes fraction',
            fontsize=12, color='green', fontweight='bold', alpha=0.6)

plt.tight_layout()
save_path2 = os.path.join(FIGURE_DIR, "inference_speed_vs_size.png")
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"📈 속도 vs 크기 버블 차트 저장: {save_path2}")
plt.close()

print(f"\n🎉 실내 모델 벤치마크 완료!")
