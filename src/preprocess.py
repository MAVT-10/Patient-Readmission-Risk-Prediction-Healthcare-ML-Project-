from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "readmission_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_readmission_data.csv"

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Replace ? with actual missing values
df = df.replace("?", pd.NA)

# Drop duplicate patient records if present
if "patient_nbr" in df.columns:
    df = df.drop_duplicates(subset=["patient_nbr"])
    print("Shape after dropping duplicate patients:", df.shape)

# Drop columns that are usually not useful for prediction
columns_to_drop = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty"
]

existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_columns_to_drop)

# Create binary target:
# 1 = readmitted within 30 days
# 0 = otherwise
df["readmitted_binary"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

print("\nValue counts for readmitted_binary:")
print(df["readmitted_binary"].value_counts())

print("\nMissing values after replacing '?':")
print(df.isnull().sum().sort_values(ascending=False).head(15))

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")