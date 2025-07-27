FROM python:3.12-slim
WORKDIR /app
COPY src/ ./src
RUN pip install pandas scikit-learn xgboost joblib
ENTRYPOINT ["python", "src/main.py"]
