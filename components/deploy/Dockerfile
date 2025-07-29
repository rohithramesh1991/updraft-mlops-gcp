FROM python:3.10-slim
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install google-cloud-aiplatform

COPY src/ ./src

ENTRYPOINT ["python", "-u", "src/main.py"]
