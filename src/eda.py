import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

print("\n========== STATISTICS ==========")
print(df.describe())

print("\n========== CROP DISTRIBUTION ==========")
print(df["label"].value_counts())

# Create plots folder
os.makedirs("plots", exist_ok=True)

# -----------------------------
# 1. Crop distribution
# -----------------------------
plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    y="label",
    order=df["label"].value_counts().index
)

plt.title("Crop Class Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig("plots/crop_distribution.png")
plt.show()


# -----------------------------
# 2. Correlation matrix
# -----------------------------
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.show()


# -----------------------------
# 3. Feature distributions
# -----------------------------
numeric_columns = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.tight_layout()

    plt.savefig(f"plots/{column}_distribution.png")
    plt.show()


# -----------------------------
# 4. Boxplots
# -----------------------------
plt.figure(figsize=(12, 7))

sns.boxplot(
    data=df[numeric_columns]
)

plt.title("Feature Boxplots")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("plots/feature_boxplots.png")
plt.show()
