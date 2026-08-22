import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("SriLanka_Tourism_Dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nColumns:")
print(df.columns.tolist())

# ==========================
# Label Encoding
# ==========================

district_encoder = LabelEncoder()
category_encoder = LabelEncoder()
budget_encoder = LabelEncoder()
place_encoder = LabelEncoder()

df["district"] = district_encoder.fit_transform(df["district"])
df["category"] = category_encoder.fit_transform(df["category"])
df["budget_level"] = budget_encoder.fit_transform(df["budget_level"])
df["place_name"] = place_encoder.fit_transform(df["place_name"])

print("\n✅ Categorical columns encoded successfully!")

print(df.head())

# ==========================
# Features & Target
# ==========================

X = df[
    [
        "district",
        "category",
        "budget_level",
        "budget_min_LKR",
        "budget_max_LKR",
        "rating"
    ]
]

y = df["place_name"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n✅ Dataset Split Completed")
print("Training Data :", len(X_train))
print("Testing Data  :", len(X_test))

# ==========================
# KNN Model
# ==========================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== KNN MODEL ==========")
print("Accuracy :", accuracy)

print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# ==========================
# Classification Report
# ==========================

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================
# Decision Tree Model
# ==========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n========== DECISION TREE MODEL ==========")
print("Accuracy :", dt_accuracy)

# ==========================
# Random Forest Model
# ==========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n========== RANDOM FOREST MODEL ==========")
print("Accuracy :", rf_accuracy)

# ==========================
# Save Best Model (KNN)
# ==========================

joblib.dump(knn, "best_model.pkl")

# ==========================
# Save Label Encoders
# ==========================

encoders = {
    "district": district_encoder,
    "category": category_encoder,
    "budget_level": budget_encoder,
    "place_name": place_encoder
}

joblib.dump(encoders, "encoders.pkl")

print("\n✅ Best model saved successfully!")
print("✅ Encoders saved successfully!")