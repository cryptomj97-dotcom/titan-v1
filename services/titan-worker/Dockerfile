FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*

COPY titan_3.0/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titan_3.0/backtest ./backtest
COPY titan_3.0/strategies ./strategies
COPY titan_3.0/core ./core

CMD ["python", "-m", "backtest.worker"]
