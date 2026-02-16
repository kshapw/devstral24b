FROM python:3.11-slim

RUN useradd -m -r appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 2024

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2024"]
