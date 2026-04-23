import os, sys, joblib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score

base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

# 2. 이제 임포트
from data.routes_indoor import load_processed_data

def visualize_accuracy_compare():
    _, X_test, _, y_test = load_processed_data()
    model_dir = "../../../models/indoor/"
    
    acc_results = {}
    for f in os.listdir(model_dir):
        if f.endswith('.pkl'):
            model = joblib.load(os.path.join(model_dir, f))
            acc_results[f] = accuracy_score(y_test, model.predict(X_test))

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(acc_results.keys()), y=list(acc_results.values()), palette='magma')
    plt.title('Final Accuracy Comparison')
    plt.ylim(0, 1.1)
    for i, v in enumerate(acc_results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.xticks(rotation=15)
    plt.show()

if __name__ == "__main__": visualize_accuracy_compare()