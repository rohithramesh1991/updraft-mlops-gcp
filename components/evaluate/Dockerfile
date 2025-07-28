FROM python:3.10-slim
WORKDIR /app

# Install ALL dependencies, including BigQuery and db-dtypes
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn joblib google-cloud-bigquery db-dtypes

COPY src/ ./src

ENTRYPOINT ["python", "-u", "src/main.py"]

