import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import json

# ---- Fix paths dynamically ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'clean_flights.csv')
model_dir = os.path.join(BASE_DIR, 'model')

os.makedirs(model_dir, exist_ok=True)

# ---- Load data ----
print("Loading clean data...")
df = pd.read_csv(data_path)

# ✅ REDUCE DATA (IMPORTANT)
df = df.sample(200000, random_state=42)

print("Shape:", df.shape)

# ---- Clean Data ----
df = df.dropna()
df = df.drop_duplicates()

# ---- Feature Engineering ----
print("\nFeature Engineering...")

df['DEP_HOUR'] = df['DEP_DELAY'].apply(lambda x: max(0, min(23, int(x/60))))
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)

# ---- Save original categories BEFORE encoding ----
airline_categories = pd.Categorical(df['AIRLINE_CODE']).categories
origin_categories = pd.Categorical(df['ORIGIN']).categories

# ---- Convert to numeric ----
df['AIRLINE_CODE'] = df['AIRLINE_CODE'].astype('category').cat.codes
df['ORIGIN'] = df['ORIGIN'].astype('category').cat.codes
df['DEST'] = df['DEST'].astype('category').cat.codes

# ---- Save mappings ----
airline_mapping = dict(enumerate(airline_categories))
origin_mapping = dict(enumerate(origin_categories))

with open(os.path.join(model_dir, 'airline_mapping.json'), 'w') as f:
    json.dump(airline_mapping, f)

with open(os.path.join(model_dir, 'origin_mapping.json'), 'w') as f:
    json.dump(origin_mapping, f)

# ---- Features & Target ----
features = [
    'AIRLINE_CODE', 'ORIGIN', 'DEST',
    'MONTH', 'DAY_OF_WEEK', 'DEP_DELAY',
    'DEP_HOUR', 'IS_WEEKEND'
]

target = 'IS_DELAYED'

X = df[features]
y = df[target]

print("Features shape:", X.shape)
print("Target distribution:")
print(y.value_counts())

# ---- Split data ----
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ❌ REMOVE SMOTE (important)

# ---- Train Model ----
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=100,     # ✅ reduced
    max_depth=12,         # ✅ reduced
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',   # ✅ handles imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained!")

# ---- Evaluate ----
print("\nEvaluating...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---- Save model ----
print("\nSaving model...")

model_path = os.path.join(model_dir, 'flight_delay_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

accuracy_data = {"model": "Optimized Random Forest", "accuracy": round(accuracy * 100, 2)}

with open(os.path.join(model_dir, 'rf_accuracy.json'), 'w') as f:
    json.dump(accuracy_data, f)

print("\n✅ Model saved successfully!")
print("🎉 Fast + accurate model ready!")

