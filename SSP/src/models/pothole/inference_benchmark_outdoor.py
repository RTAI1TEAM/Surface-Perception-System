import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score, recall_score, average_precision_score

# 0. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../../data/processed/outdoor"))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../../models/pothole"))
FIGURE_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../../reports/figures/final_comparison"))
os.makedirs(FIGURE_DIR, exist_ok=True)

DATASET_NAME = "v3_s2"
N_REPEATS = 1000

# 1. 테스트 데이터 로드
print("Loading test data...")
test_df = pd.read_csv(os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv"))
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. 모델 정의
model_configs = {
    "XGBoost (Tuned)": {"model_file": "XGB1_best_model.pkl", "scaler_file": None, "is_dict": True, "threshold": 0.5},
    "Random Forest (Class+Grid)": {"model_file": "rf_class_grid_v3_s2.pkl", "scaler_file": None, "is_dict": False, "threshold": 0.4528},
    "SVM (ClassWeight+Grid)": {"model_file": "svm_class_grid_v3_s2.pkl", "scaler_file": "svm_scaler_v3_s2.pkl", "is_dict": False, "threshold": 0.2083},
    "Logistic (ClassWeight+Grid)": {"model_file": "logistic_class_grid_v3_s2.pkl", "scaler_file": "logistic_scaler_v3_s2.pkl", "is_dict": False, "threshold": 0.5152},
}

# 3. 벤치마크 실행
results = []
for name, config in model_configs.items():
    model_path = os.path.join(MODEL_DIR, config["model_file"])
    if not os.path.exists(model_path): continue
    
    raw = joblib.load(model_path)
    model = raw["model"] if config["is_dict"] else raw
    scaler = joblib.load(os.path.join(MODEL_DIR, config["scaler_file"])) if config["scaler_file"] else None
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if config["scaler_file"]: model_size_mb += os.path.getsize(os.path.join(MODEL_DIR, config["scaler_file"])) / (1024 * 1024)
    
    X_input = scaler.transform(X_test) if scaler else X_test.values
    threshold = config["threshold"]
    
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        model.predict_proba(X_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    y_probs = model.predict_proba(X_input)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    results.append({
        "Model": name, "Avg_ms": np.mean(times), "Size_MB": model_size_mb,
        "F1": f1_score(y_test, y_pred), "Recall": recall_score(y_test, y_pred),
        "PR_AUC": average_precision_score(y_test, y_probs)
    })
    print(f"DONE: {name}")

df = pd.DataFrame(results)
csv_path = os.path.join(FIGURE_DIR, "outdoor_inference_benchmark.csv")
df.to_csv(csv_path, index=False)
print(f"Success: {csv_path}")
