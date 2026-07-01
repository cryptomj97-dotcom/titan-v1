FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev curl && rm -rf /var/lib/apt/lists/*

COPY titan_3.0/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titan_3.0/api ./api
COPY titan_3.0/core ./core

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
