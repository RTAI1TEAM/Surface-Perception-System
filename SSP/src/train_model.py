import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed/indoor/indoor_train_features.csv')

X = df.drop(['series_id', 'group_id', 'surface', 'surface_encoded'], axis=1)
y = df['surface_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, pred))