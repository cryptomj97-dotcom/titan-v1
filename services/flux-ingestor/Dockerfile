FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY titan_3.0/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY titan_3.0/realtime_flux ./realtime_flux
COPY titan_3.0/core ./core
COPY titan_3.0/config ./config

# Run the Flux Ingestor
CMD ["python", "-m", "realtime_flux.main"]
