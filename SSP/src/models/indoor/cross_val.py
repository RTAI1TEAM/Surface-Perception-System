import os, sys, joblib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import cross_val_score

# 1. 절대 경로 설정 (가장 확실한 방법)
base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

from data.routes_indoor import load_processed_data

def visualize_cv():
    # 데이터 로드 (여기도 필요하다면 인자로 경로 전달)
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. 모델 경로를 base_path(src) 기준으로 절대 경로화
    # src 폴더와 같은 위치에 있는 models 폴더를 찾아갑니다.
    model_dir = os.path.abspath(os.path.join(base_path, "../models/indoor/"))
    
    print(f"🔎 모델을 찾는 중: {model_dir}") # 확인용 출력
    
    if not os.path.exists(model_dir):
        print("❌ 모델 디렉토리가 존재하지 않습니다!")
        return

    cv_means, cv_stds, names = [], [], []
    
    # 디렉토리 내 파일 목록 확인
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    if not files:
        print("❌ 해당 폴더에 .pkl 모델 파일이 하나도 없습니다!")
        return

    for f in files:
        name = f.replace('.pkl', '')
        model = joblib.load(os.path.join(model_dir, f))
        
        print(f"🔄 {name} 모델 교차 검증 중...")
        scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
        names.append(name)

    # 3. 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.errorbar(names, cv_means, yerr=cv_stds, fmt='o', linestyle='-', color='r', ecolor='lightgray', capsize=5)
    plt.title('5-Fold Cross-Validation Accuracy (Mean & Std Dev)', fontsize=14)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=15) # 모델 이름이 겹치지 않게 회전
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    visualize_cv()