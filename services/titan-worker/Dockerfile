FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas redis aiohttp

COPY src/ ./src/

CMD ["python", "src/main.py"]
