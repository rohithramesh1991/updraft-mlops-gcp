FROM python:3.10-slim
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pandas scikit-learn joblib google-cloud-bigquery db-dtypes

COPY src/ ./src
ENTRYPOINT ["python", "-u", "src/main.py"]
