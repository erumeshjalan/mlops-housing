import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Load trained model
model = joblib.load("models/sklearn_model.joblib")

# Make predictions
y_pred = model.predict(X)

# Evaluate model
r2 = r2_score(y, y_pred)
print(f"[PREDICT] R² Score: {r2:.6f}")
