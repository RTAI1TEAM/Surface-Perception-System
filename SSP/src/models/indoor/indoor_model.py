import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# 6개 분류 모델 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier # 설치 필요: pip install xgboost

# 1. 데이터 로드

# 상대 경로 방식
df = pd.read_csv('../../data/processed/indoor/indoor_train_features.csv')

# 2. 타겟(레이블) 컬럼 설정
# 파일 내에 'surface' 컬럼이 있다고 하셨으므로 이를 사용합니다. 
# 만약 이름이 조금 다르다면 아래 리스트에서 수정해주세요.
target_col = 'surface_encoded' if 'surface_encoded' in df.columns else df.columns[-1]

# 특징량(X)과 타겟(y) 분리
# series_id와 같은 식별용 컬럼은 학습에서 제외합니다.
# 1. 타겟(y) 설정
target_col = 'surface_encoded' # 또는 사용자가 지정한 정답 컬럼명
y = df[target_col]

# 2. 특징량(X)에서 불필요한 컬럼들 한꺼번에 제거
# series_id, group_id는 식별용이므로 빼고, target_col은 정답이므로 뺍니다.
X = df.drop(columns=['series_id', 'group_id', 'surface', target_col])

# 확인용: 남은 컬럼들이 센서 데이터(mean, std 등)만 있는지 확인
print(X.columns.tolist())

# 3. 데이터 분할 (8:2 비율, 정규화 생략)
# 클래스 비율 유지를 위해 stratify를 적용합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 6가지 모델 정의
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results_acc = {}
cms = {}

# 5. 모델 학습 및 평가 루프
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 정확도 저장
    acc = accuracy_score(y_test, y_pred)
    results_acc[name] = acc
    
    # Confusion Matrix 저장
    cms[name] = confusion_matrix(y_test, y_pred)
    print(f"✅ {name} 학습 완료 (Accuracy: {acc:.4f})")

# ✅ Decision Tree 학습 완료 (Accuracy: 0.7231)
# ✅ SVM 학습 완료 (Accuracy: 0.4829)
# ✅ Naive Bayes 학습 완료 (Accuracy: 0.4331)
# ✅ Random Forest 학습 완료 (Accuracy: 0.8268)
# ✅ Logistic Regression 학습 완료 (Accuracy: 0.4948)
# ✅ XGBoost 학습 완료 (Accuracy: 0.8648)

# [추가] 1. 교차 검증 (Cross Validation)
print("\n--- 5-Fold 교차 검증 결과 ---")
cv_results = {}
for name, model in models.items():
    # cv=5는 데이터를 5등분하여 5번 검증한다는 뜻입니다.
    # 전체 데이터 X, y를 사용하거나 학습 데이터 X_train, y_train을 사용합니다.
    scores = cross_val_score(model, X, y, cv=5)
    cv_results[name] = scores.mean()
    print(f"📊 {name} CV Average Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# [작업 1] 5-Fold 교차 검증 수행 및 시각화
print("\n--- 5-Fold 교차 검증 및 시각화 시작 ---")
cv_means = []
cv_stds = []

for name, model in models.items():
    # 전체 데이터 X, y를 사용하여 모델의 일반화 성능 검증
    scores = cross_val_score(model, X, y, cv=5)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"📊 {name}: CV Mean = {scores.mean():.4f}")

# CV 결과 시각화
plt.figure(figsize=(12, 6))
plt.errorbar(models.keys(), cv_means, yerr=cv_stds, fmt='o', linestyle='-', color='r', ecolor='lightgray', capsize=5)
plt.title('5-Fold Cross-Validation Accuracy (Mean & Std Dev)', fontsize=15)
plt.ylabel('Accuracy')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# [작업 2] 하이퍼파라미터 튜닝 (GridSearchCV)
# 1. Random Forest 튜닝
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf_acc = accuracy_score(y_test, grid_rf.best_estimator_.predict(X_test))

# 2. XGBoost 튜닝
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6]
}
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, cv=3, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb_acc = accuracy_score(y_test, grid_xgb.best_estimator_.predict(X_test))

# [작업 3] 튜닝 전후 성능 변화 확인
tuning_results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Before Tuning': [results_acc['Random Forest'], results_acc['XGBoost']],
    'After Tuning': [best_rf_acc, best_xgb_acc]
})

print("\n--- 하이퍼파라미터 튜닝 결과 ---")
print(tuning_results)

# 변화값 시각화
tuning_results.set_index('Model').plot(kind='bar', figsize=(10, 6), color=['lightgray', 'skyblue'])
plt.title('Accuracy Improvement After Hyperparameter Tuning', fontsize=15)
plt.ylabel('Accuracy')
plt.ylim(results_acc['Random Forest'] - 0.05, 1.0)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

# 1. 튜닝했는데 왜 정확도가 떨어질까? (충격 방지 가이드)
# 탐색 범위의 한계: GridSearchCV에 넣은 파라미터 후보들 자체가 현재 모델의 기본값(Default)보다 안 좋은 조합들만 있었을 가능성이 큽니다.

# 데이터 분할의 우연성: 기본 모델은 전체 학습 데이터를 다 썼는데, 튜닝 시 cv=3 등을 쓰면서 데이터를 쪼개 학습하다 보니 특정 폴드에서 성능이 낮게 측정되었을 수 있습니다.

# 과적합 방지 기법의 부작용: 튜닝 과정에서 모델을 단순화하는 파라미터가 선택되면, 학습 데이터에 대한 정확도는 낮아지지만 실제로는 더 **'일반화'**된 좋은 모델이 된 것일 수도 있습니다.


# [추가] 2. Feature Importance 시각화 (Tree 기반 모델)
# 모델 자체의 feature_importances_ 속성을 사용합니다.
# 가장 성능이 좋은 XGBoost와 Random Forest의 중요도를 뽑아봅니다.

def plot_all_feature_importances(models, features):
    # 트리 기반 모델만 추출 (Random Forest, XGBoost 등)
    tree_models = {k: v for k, v in models.items() if hasattr(v, 'feature_importances_')}
    
    if not tree_models:
        print("시각화할 트리 기반 모델이 없습니다.")
        return

    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1: axes = [axes] # 모델이 하나일 때 예외 처리

    for i, (name, model) in enumerate(tree_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:] # 상위 15개
        
        axes[i].barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        axes[i].set_yticks(range(len(indices)))
        axes[i].set_yticklabels([features[idx] for idx in indices])
        axes[i].set_title(f'Feature Importances: {name}', fontsize=15)
        axes[i].set_xlabel('Relative Importance')

    plt.tight_layout()
    plt.show()

# 실행
plot_all_feature_importances(models, X.columns)

# 6. 시각화 1: Confusion Matrix Subplots (2x3 레이아웃)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f"{name}\n(Acc: {results_acc[name]:.4f})", fontsize=14)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 7. 시각화 2: 정확도 비교 바 차트
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results_acc.keys()), y=list(results_acc.values()), palette='magma')
plt.title('Accuracy Comparison of 6 Classification Models', fontsize=16)
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)

# 막대 위에 정확도 값 표시
for i, v in enumerate(results_acc.values()):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

from sklearn.ensemble import VotingClassifier

# 성능이 좋았던 모델들을 선별하여 앙상블 구성
voting_model = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost'])
    ],
    voting='soft' # 확률 기반 투표
)

voting_model.fit(X_train, y_train)
y_pred_voting = voting_model.predict(X_test)
voting_acc = accuracy_score(y_test, y_pred_voting)

print(f"🚀 Voting Ensemble Accuracy: {voting_acc:.4f}")
# 🚀 Voting Ensemble Accuracy: 0.8596 --> 정확도 더 떨어짐

# 1. 테스트 결과 중 정확도가 가장 높은 모델 찾기
# results_acc는 이전에 구한 {모델명: 정확도} 딕셔너리입니다.
best_model_name = max(results_acc, key=results_acc.get)
best_model_obj = models[best_model_name]

print(f"\n🏆 최종 선정된 모델: {best_model_name} (정확도: {results_acc[best_model_name]:.4f})")

# 2. 모델 저장 경로 설정
# PDF의 SSP 프로젝트 구조에 맞춰 app/models/ 폴더 등이 있다면 그곳에 저장하는 것이 좋습니다.
save_path = '../../models/indoor/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_filename = os.path.join(save_path, 'best_indoor_model.pkl')

# 3. 모델 덤프(Dump)
joblib.dump(best_model_obj, model_filename)

print(f"📁 모델 저장 완료: {model_filename}")

# 4. (참고) 나중에 불러올 때는?
# loaded_model = joblib.load(model_filename)