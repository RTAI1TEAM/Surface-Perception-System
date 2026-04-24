import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shutil
from sklearn.metrics import f1_score

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
MODEL_DIR = os.path.join(BASE_DIR, "models", "indoor")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.abspath(os.path.join(current_dir, "../../")) 
if project_src not in sys.path:
    sys.path.append(project_src)
from data.routes_indoor import load_processed_data

def select_and_save_final_best():
    # 2. 데이터 로드
    print("🚀 최종 대표 모델 선발을 위한 데이터 로드 중...")
    _, X_test, _, y_test = load_processed_data()

    # 모든 튜닝 결과 파일 리스트
    tuning_methods = ["Grid", "Random", "Bayesian"]
    algorithms = ["DecisionTree", "RandomForest", "XGBoost"]
    
    results = []

    # 3. 모든 모델 성능 평가
    print("\n--- 전수 조사 시작 ---")
    for algo in algorithms:
        best_score_for_algo = -1
        best_model_path = ""
        best_method_name = ""

        for method in tuning_methods:
            file_name = f"{method}_Tuned_{algo}.pkl"
            path = os.path.join(MODEL_DIR, file_name)
            
            if os.path.exists(path):
                model = joblib.load(path)
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average='macro')
                
                results.append({
                    "Algorithm": algo,
                    "Method": method,
                    "F1_Score": score
                })

                # 알고리즘별 최고점 갱신 확인
                if score > best_score_for_algo:
                    best_score_for_algo = score
                    best_model_path = path
                    best_method_name = method
            else:
                print(f"⚠️ 파일 없음: {file_name}")

        # 4. 알고리즘별 최종 베스트 모델 저장 (복사)
        if best_model_path:
            final_save_path = os.path.join(MODEL_DIR, f"Final_Best_{algo}.pkl")
            shutil.copy(best_model_path, final_save_path)
            print(f"👑 {algo}의 최종 승자: {best_method_name} (F1: {best_score_for_algo:.4f})")
            print(f"   -> {final_save_path}로 저장 완료")

    # 5. 시각화 (어떤 튜닝 방식이 승리했는지 강조)
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df_res, x='Algorithm', y='F1_Score', hue='Method', palette='viridis')
    
    # 수치 및 승자 표시
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Final Champion Selection: Tuning Method Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('F1-Score (Macro)')
    plt.legend(title="Tuning Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "final_best_selection.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    select_and_save_final_best()