from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_readmission_data.csv"
MODEL_PATH = BASE_DIR / "model"
MODEL_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

selected_features = [
    "age",
    "gender",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "num_procedures",
    "number_diagnoses"
]

target = "readmitted_binary"

demo_df = df[selected_features + [target]].copy()

X = demo_df[selected_features].copy()
y = demo_df[target].copy()

label_encoders = {}

for col in X.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_PATH / "demo_logistic_model.pkl")
joblib.dump(scaler, MODEL_PATH / "demo_scaler.pkl")
joblib.dump(label_encoders, MODEL_PATH / "demo_label_encoders.pkl")
joblib.dump(selected_features, MODEL_PATH / "demo_feature_names.pkl")

print("\nDemo model artifacts saved successfully.")