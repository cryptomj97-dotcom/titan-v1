FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*

COPY titan_3.0/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titan_3.0/monitoring ./monitoring
COPY titan_3.0/core ./core

EXPOSE 3000 9090

CMD ["python", "-m", "monitoring.dashboard"]
