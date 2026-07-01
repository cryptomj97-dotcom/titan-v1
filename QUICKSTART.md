# TITAN 3.0 Microservices - Quick Start Guide

## Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+
- At least 4GB RAM available

## Quick Start (5 Minutes)

### 1. Setup Environment
```bash
cd /workspaces/titan-v1
cp .env.example .env
```

### 2. Launch All Services
```bash
docker-compose up -d --build
```

### 3. Verify Services
```bash
docker-compose ps
```
All services should show "Up (healthy)" status.

### 4. Access Points
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

### 5. View Live Logs
```bash
docker-compose logs -f
```

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐
│  Flux Ingestor  │────▶│    Redis     │
│  (Binance WS)   │     │  (Pub/Sub)   │
└─────────────────┘     └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌──────▼──────┐  ┌─────▼─────┐
        │ Titan Core│   │Titan Executor│  │Titan Worker│
        │(Strategy) │   │  (Orders)    │  │(Backtest) │
        └─────┬─────┘   └──────┬──────┘  └─────┬─────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                        ┌──────▼──────┐
                        │  Titan API  │
                        │  (Gateway)  │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Frontend  │
                        │  (WebSocket)│
                        └─────────────┘
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| flux-ingestor | - | Real-time Binance WebSocket ingestion |
| titan-core | - | Strategy engine, regime detection, signals |
| titan-executor | - | Order execution, risk management |
| titan-api | 8000 | REST API + WebSocket gateway |
| titan-worker | - | Background jobs (backtesting) |
| titan-monitor | 9090/3000 | Prometheus + Grafana monitoring |
| redis | 6379 | Message broker & state store |
| timescaledb | 5432 | Persistent time-series database |

## Common Commands

### Stop All Services
```bash
docker-compose down
```

### Restart a Specific Service
```bash
docker-compose restart titan-api
```

### View Specific Service Logs
```bash
docker-compose logs -f flux-ingestor
```

### Scale Worker Service
```bash
docker-compose up -d --scale titan-worker=3
```

## Testing the System

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

### 2. Test WebSocket Connection
Open browser console or use wscat:
```bash
wscat -c ws://localhost:8000/ws/market
```

### 3. Check Redis Data
```bash
docker-compose exec redis redis-cli KEYS "*"
docker-compose exec redis redis-cli HGETALL latest_prices
```

## Troubleshooting

### Services Won't Start
```bash
# Check logs
docker-compose logs titan-core

# Rebuild images
docker-compose build --no-cache

# Check resource usage
docker stats
```

### Connection Issues
```bash
# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check network
docker-compose exec titan-api ping redis
```

## Next Steps

1. **Configure API Keys**: Edit `.env` with your exchange credentials for live trading
2. **Customize Strategies**: Modify `titan-core/src/main.py` for your trading logic
3. **Set Up Alerts**: Configure Grafana alerts in the dashboard
4. **Monitor Performance**: Watch metrics at http://localhost:3000

## Security Notes

- Never commit `.env` file to version control
- Change default passwords before production deployment
- Use SSL/TLS for production API endpoints
- Enable authentication for Grafana in production

---

**Status**: ✅ All services implemented and ready for deployment
**Version**: 3.0.0
**License**: MIT
