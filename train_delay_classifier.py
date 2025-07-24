import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("vehicle_with_features.csv")

# Target label: is_delayed = 1 if delay > 10 mins
df['is_delayed'] = (df['TimeDiffToStop_min'] > 10).astype(int)

# Features and target
features = ['Speed', 'DistanceToNearestStop_m', 'StoppedDuration', 'Is_Night', 'Hour', 'DayOfWeek']
X = df[features]
y = df['is_delayed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Random Forest Performance:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# Save model to .pkl
joblib.dump(model, "delay_classifier.pkl")
print("\nModel saved as delay_classifier.pkl")
