FROM python:3.11-slim

WORKDIR /app

# Install system compilation packages for psycopg2 compliance
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache requirements installation
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    psycopg2-binary \
    scikit-learn \
    numpy

COPY app.py seed_data.py /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
