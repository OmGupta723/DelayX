import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'clean_flights.csv')
model_dir = os.path.join(BASE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)

print("Loading clean data...")
df = pd.read_csv(data_path)
print("Shape:", df.shape)

df = df.dropna()
df = df.drop_duplicates()

df['DEP_HOUR'] = df['DEP_DELAY'].apply(lambda x: max(0, min(23, int(x/60))))
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)

airline_categories = pd.Categorical(df['AIRLINE_CODE']).categories
origin_categories = pd.Categorical(df['ORIGIN']).categories

df['AIRLINE_CODE'] = df['AIRLINE_CODE'].astype('category').cat.codes
df['ORIGIN'] = df['ORIGIN'].astype('category').cat.codes
df['DEST'] = df['DEST'].astype('category').cat.codes

airline_mapping = dict(enumerate(airline_categories))
origin_mapping = dict(enumerate(origin_categories))

with open(os.path.join(model_dir, 'airline_mapping.json'), 'w') as f:
    json.dump(airline_mapping, f)
with open(os.path.join(model_dir, 'origin_mapping.json'), 'w') as f:
    json.dump(origin_mapping, f)

features = ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK', 'DEP_DELAY', 'DEP_HOUR', 'IS_WEEKEND']
target = 'IS_DELAYED'

X = df[features]
y = df[target]

# SVM is slow on large data — sample 50,000 rows max
MAX_ROWS = 50_000
if len(df) > MAX_ROWS:
    print(f"\n⚠️ Dataset too large for SVM. Sampling {MAX_ROWS:,} rows...")
    df = df.sample(n=MAX_ROWS, random_state=42)
    X = df[features]
    y = df[target]

print("Target distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open(os.path.join(model_dir, 'svm_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("\nTraining SVM model (may take a few minutes)...")
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 SVM Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

with open(os.path.join(model_dir, 'svm_flight_delay_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
    accuracy_data = {"model": "SVM", "accuracy": round(accuracy * 100, 2)}
with open(os.path.join(model_dir, 'svm_accuracy.json'), 'w') as f:
    json.dump(accuracy_data, f)

print("\ n✅ SVM Model saved!")