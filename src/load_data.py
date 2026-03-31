from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "readmission_data.csv"

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget column value counts (readmitted):")
print(df["readmitted"].value_counts())

print("\nUnique values in readmitted:")
print(df["readmitted"].unique())