from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_readmission_data.csv"

df = pd.read_csv(DATA_PATH)

print("Loaded cleaned data:", df.shape)

# Drop original target column
df = df.drop(columns=["readmitted"])

# Separate features and target
X = df.drop(columns=["readmitted_binary"])
y = df["readmitted_binary"]

FEATURES_PATH = BASE_DIR / "model"
FEATURES_PATH.mkdir(exist_ok=True)

joblib.dump(X.columns.tolist(), FEATURES_PATH / "feature_names.pkl")

# Encode categorical variables
label_encoders = {}

for col in X.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("\nAfter encoding:")
print(X.head())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTarget distribution in train:")
print(y_train.value_counts(normalize=True))

print("\nTarget distribution in test:")
print(y_test.value_counts(normalize=True))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train improved logistic regression
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

MODEL_PATH = BASE_DIR / "model"
MODEL_PATH.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH / "logistic_model.pkl")
joblib.dump(scaler, MODEL_PATH / "scaler.pkl")
joblib.dump(label_encoders, MODEL_PATH / "label_encoders.pkl")
joblib.dump(X.columns.tolist(), MODEL_PATH / "feature_names.pkl")

print("\nModel, scaler, and encoders saved.")

from sklearn.ensemble import RandomForestClassifier

print("\n--- RANDOM FOREST MODEL ---")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

import matplotlib.pyplot as plt
import pandas as pd

# Feature importance
feature_importances = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": feature_importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Plot top 10
top10 = importance_df.head(10)

plt.figure()
plt.barh(top10["feature"], top10["importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(BASE_DIR / "outputs" / "feature_importance.png")
plt.show()