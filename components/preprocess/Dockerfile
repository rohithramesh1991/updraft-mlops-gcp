FROM python:3.10-slim
WORKDIR /app
COPY src/ ./src
RUN pip install --upgrade pip && \
    pip install pandas 
ENTRYPOINT ["python", "src/main.py"]
