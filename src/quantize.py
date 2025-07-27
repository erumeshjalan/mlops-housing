import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_absolute_error

# Load trained model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized parameters
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "models/unquant_params.joblib")

# ----- Quantization -----

# Step 1: Get min and max of weights
min_val = weights.min()
max_val = weights.max()

# Step 2: Calculate scale
scale = (max_val - min_val) / 255.0

# Step 3: Quantize weights to uint8
quantized_weights = np.clip(np.round((weights - min_val) / scale), 0, 255).astype(np.uint8)

# Step 4: Dequantize
dequantized_weights = quantized_weights.astype(np.float32) * scale + min_val

# Bias is preserved as-is (float)
# Reason: Single constant → minimal memory → precision matters more

# Save quantized parameters
quant_params = {
    "weights": quantized_weights,
    "bias": bias,  # Preserved as float32
    "scale": scale,
    "min": min_val
}
joblib.dump(quant_params, "models/quant_params.joblib")

# ----- Reconstruct and Evaluate -----

class ReconstructedModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Use dequantized weights and original bias
reconstructed_model = ReconstructedModel(dequantized_weights, bias)
y_pred = reconstructed_model.predict(X)

# Evaluate performance
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(weights, dequantized_weights)

# ---- Final Output ----
print("==== Quantization Summary ====")
print(f"Original weight range: [{min_val:.6f}, {max_val:.6f}]")
print(f"Scale used: {scale:.8f}")
print(f"Quantization Error (MAE): {mae:.8f}")
print(f"Bias (preserved): {bias:.6f}")
print(f"[QUANTIZED] R² Score: {r2:.6f}")
print("Bias preserved as float32 due to minimal impact and precision retention.")
print("Quantization complete.")
