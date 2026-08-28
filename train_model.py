import csv
import os
# pyrefly: ignore [missing-import]
import pickle
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from sklearn.metrics import classification_report

# 1. Load the landmarks.csv dataset
X = []
y = []
try:
    with open("dataset/landmarks.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row

        for row in reader:
            if not row:
                continue

            # Label is the last column, strip whitespace
            label = row[-1].strip()
            y.append(label)

            # 63 coordinate features (21 landmarks x 3 coords) — all columns except last
            features = [float(val) for val in row[:-1]]
            X.append(features)
except FileNotFoundError:
    print("dataset/landmarks.csv not found! Run generate_synthetic_data.py first.")
    exit()

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("No data found in landmarks.csv!")
    exit()

print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
print(f"Classes: {sorted(set(y))}")
print(f"Samples per class:")
for cls in sorted(set(y)):
    print(f"  {cls}: {np.sum(y == cls)}")

# 2. Split into train/test (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Train RandomForest classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=1
)
model.fit(X_train, y_train)

# 4. Evaluate — honest accuracy on held-out test set
test_accuracy = model.score(X_test, y_test)
print(f"\n--- Test Set Accuracy: {test_accuracy * 100:.2f}% ---")

# 5. Cross-validation for true generalization estimate (commented out to reduce output)
    # cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
    # print(f"--- 5-Fold Cross-Validation: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%) ---")

# 6. Per-class classification report
print("\nPer-Class Classification Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save model
os.makedirs("models", exist_ok=True)
with open("models/sign_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/sign_model.pkl")