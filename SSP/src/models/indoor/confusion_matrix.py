import os, sys, joblib, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

from data.routes_indoor import load_processed_data

def visualize_cm():
    _, X_test, _, y_test = load_processed_data()
    
    # 모델 경로도 src와 같은 위치에 있는 models 폴더를 바라보도록 설정
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/indoor"))
    
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files: 
        print("모델 파일이 없습니다!"); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, f in enumerate(files[:6]):
        model = joblib.load(os.path.join(model_dir, f))
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"CM: {f}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": visualize_cm()