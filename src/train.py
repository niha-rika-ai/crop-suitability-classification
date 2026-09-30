import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. BASIC CLEANING
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicates if present
df = df.drop_duplicates()


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = df[features]
y = df["label"]


# ==========================================
# 4. ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    )
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = {}

best_model_name = None
best_model = None
best_f1 = -1


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Automatically retain model with highest weighted F1
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results).T

print("\n========== MODEL COMPARISON ==========")
print(results_df)


# ==========================================
# 9. FINAL MODEL
# ==========================================

print("\nSelected model:", best_model_name)

final_predictions = best_model.predict(X_test)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig("plots/confusion_matrix.png")

plt.show()


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/crop_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("\nModel saved successfully.")


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "models/model_comparison.csv"
)

print("\nResults saved to models/model_comparison.csv")