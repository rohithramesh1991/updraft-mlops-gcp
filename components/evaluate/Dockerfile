FROM python:3.10-slim
WORKDIR /app

# Install ALL dependencies, including BigQuery and db-dtypes
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn==1.0.2 xgboost==1.5.1 joblib==1.1.0 google-cloud-bigquery db-dtypes seaborn matplotlib

COPY src/ ./src

ENTRYPOINT ["python", "-u", "src/main.py"]

