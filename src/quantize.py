import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load original model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized params
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "models/unquant_params.joblib")

# Quantization
min_w, max_w = weights.min(), weights.max()
scale = (max_w - min_w) / 255.0
quantized_weights = np.clip(np.round((weights - min_w) / scale), 0, 255).astype(np.uint8)

# Dequantization
dequantized_weights = quantized_weights.astype(np.float32) * scale + min_w
dequantized_bias = bias  # Single value, kept as float

# Save quantized
quant_params = {"weights": quantized_weights, "bias": dequantized_bias, "scale": scale, "min": min_w}
joblib.dump(quant_params, "models/quant_params.joblib")

# Rebuild model with dequantized weights
class ReconstructedModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Evaluate dequantized model
data = fetch_california_housing()
X, y = data.data, data.target
reconstructed_model = ReconstructedModel(dequantized_weights, dequantized_bias)
y_pred = reconstructed_model.predict(X)
r2 = r2_score(y, y_pred)

print(f"[QUANTIZED] R² Score: {r2:.6f}")
print("Quantization complete.")
