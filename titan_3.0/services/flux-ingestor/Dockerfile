FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir websockets redis aiohttp

# Copy service code
COPY src/ ./src/

# Run the Flux Ingestor
CMD ["python", "src/main.py"]
