import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Drop rows with missing target
df = df[df["Sleep Disorder"].notnull()]

# Map labels to numbers
label_map = {"None": 0, "Insomnia": 1, "Sleep Apnea": 2}
df["Sleep Disorder"] = df["Sleep Disorder"].map(label_map)

# Features and labels
X = df.drop("Sleep Disorder", axis=1)
y = df["Sleep Disorder"]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, stratify=y, test_size=0.2, random_state=42
)

# ------------------------------
# Build pipeline (scaler + model)
# ------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse data
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save trained pipeline
joblib.dump(pipeline, "sleep_pipeline.pkl")

print("✅ Training complete. Model saved as sleep_pipeline.pkl")
