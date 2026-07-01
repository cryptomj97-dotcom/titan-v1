# TITAN 3.0 Microservices Deployment Guide

## Prerequisites
- Docker & Docker Compose installed
- Git configured
- API keys for data providers (Binance, Alpha Vantage, etc.)

## Quick Start

### 1. Clone and Configure
```bash
git clone https://github.com/cryptomj97-dotcom/titan-v1.git
cd titan-v1

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

### 2. Launch the Fleet
```bash
docker-compose up -d --build
```

### 3. Verify Health
```bash
docker-compose ps
# All services should show "Up (healthy)"

# View logs
docker-compose logs -f titan-api
```

### 4. Access Services
- **API Gateway**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin_secure_change_me)
- **Prometheus**: http://localhost:9090

## Service Architecture

| Service | Port | Purpose |
|---------|------|---------|
| `titan-flux` | - | Real-time WebSocket ingestion |
| `titan-core` | - | Strategy engine, RL, Debate |
| `titan-executor` | - | Order management, Risk checks |
| `titan-api` | 8000 | REST API & WebSocket gateway |
| `titan-worker` | - | Background jobs (backtest, optimization) |
| `titan-monitor` | 3000/9090 | Grafana & Prometheus monitoring |
| `redis` | 6379 | State management & Pub/Sub |
| `timescaledb` | 5432 | Persistent time-series storage |

## Common Commands

### Restart a service
```bash
docker-compose restart titan-core
```

### Scale workers
```bash
docker-compose up -d --scale titan-worker=3
```

### View real-time logs
```bash
docker-compose logs -f --tail=100
```

### Stop all services
```bash
docker-compose down
```

### Stop and remove volumes (reset data)
```bash
docker-compose down -v
```

## Troubleshooting

### Service not starting
```bash
docker-compose logs <service-name>
```

### Database connection issues
```bash
docker-compose exec timescaledb psql -U titan -d titan_market_data
```

### Redis connection test
```bash
docker-compose exec redis redis-cli ping
```

## Production Hardening

1. **Change default passwords** in `.env`
2. **Enable SSL** for API endpoints
3. **Configure firewall** rules
4. **Set up log rotation**
5. **Monitor disk space** for volumes
6. **Enable Redis persistence** (AOF enabled by default)

## Security Notes
- Never commit `.env` to Git
- Rotate API keys regularly
- Use network policies in production
- Enable authentication for Grafana
