FROM python:3.10-slim
WORKDIR /app

# Install ALL dependencies, including BigQuery and db-dtypes
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn xgboost joblib google-cloud-bigquery db-dtypes seaborn matplotlib

COPY src/ ./src

ENTRYPOINT ["python", "-u", "src/main.py"]

