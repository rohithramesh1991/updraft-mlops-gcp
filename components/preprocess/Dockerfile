FROM python:3.12-slim
WORKDIR /app
COPY src/ ./src
RUN pip install pandas
ENTRYPOINT ["python", "src/main.py"]
