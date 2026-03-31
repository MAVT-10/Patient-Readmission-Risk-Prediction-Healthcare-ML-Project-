from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_readmission_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Cleaned shape:", df.shape)

print("\nTarget distribution:")
print(df["readmitted_binary"].value_counts(normalize=True))

# Plot target distribution
df["readmitted_binary"].value_counts().plot(kind="bar")
plt.title("Readmission Within 30 Days")
plt.xlabel("Readmitted Binary (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png")
plt.show()

# Age distribution
if "age" in df.columns:
    df["age"].value_counts().sort_index().plot(kind="bar")
    plt.title("Age Group Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png")
    plt.show()

# Gender distribution
if "gender" in df.columns:
    df["gender"].value_counts().plot(kind="bar")
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_distribution.png")
    plt.show()