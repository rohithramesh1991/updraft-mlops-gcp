FROM python:3.10-slim
WORKDIR /app
COPY src/ ./src
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn==1.0.2 xgboost==1.5.1 joblib==1.1.0 numpy==1.21.6
ENTRYPOINT ["python", "-u", "src/main.py"]
