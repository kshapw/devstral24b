FROM python:3.11-slim

# Create a non-root user
RUN useradd -m -r appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy data directory (ensure you have .dockerignore to exclude local DB files if needed)
COPY data/ ./data/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the requested port
EXPOSE 2024

# Run the application on port 2024
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2024"]
