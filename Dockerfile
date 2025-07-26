# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model into container
COPY src/ ./src/
COPY models/ ./models/

# Default command: run prediction
CMD ["python", "src/predict.py"]
