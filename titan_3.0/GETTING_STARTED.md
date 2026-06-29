# 🚀 TITAN 3.0 - Getting Started Guide

## Prerequisites
- Python 3.9+
- pip (Python package manager)
- Git

## Quick Start

### 1. Installation

```bash
cd /workspace/titan_3.0

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/master_config.yaml` to customize:
- API keys (Alpha Vantage, NewsAPI, etc.)
- Risk parameters
- Trading assets
- Backtesting periods

**Important**: Set environment variables for sensitive keys:
```bash
export ALPHA_VANTAGE_KEY="your_key_here"
export NEWS_API_KEY="your_key_here"
export SLACK_WEBHOOK="your_webhook_here"
```

### 3. Run Modes

#### A. Full System (API + Monitoring Dashboard)
```bash
python main.py
```
- API Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Monitoring Dashboard: http://localhost:8001

#### B. Backtesting Engine
```bash
python main.py --backtest
```
Generates detailed reports in `./reports/`

#### C. Train RL Models
```bash
python main.py --train
```
Saves trained models to `./models/`

#### D. Debate Simulation
```bash
python main.py --debate
```
Runs a standalone adversarial debate demonstration

### 4. Frontend War Room

Open the trading interface:
```bash
# Option 1: Direct file open
open frontend/index.html

# Option 2: Serve with Python
cd frontend
python -m http.server 3000
# Visit: http://localhost:3000
```

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TITAN 3.0 Ecosystem                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 Data Layer                                               │
│     ├─ Market Data (Yahoo, Alpha Vantage)                   │
│     ├─ Alternative Data (News, Social Media)                │
│     └─ Feature Engineering (50+ indicators)                 │
│                                                              │
│  🧠 Intelligence Layer                                       │
│     ├─ Regime Detection (TDA, HMM)                          │
│     ├─ Strategy Generator (Genetic Algorithms)              │
│     ├─ RL Agent (PPO)                                       │
│     └─ Adversarial Debate (Bull/Bear/Judge)                 │
│                                                              │
│  ⚡ Execution Layer                                          │
│     ├─ Risk Management (Kelly, VaR, Drawdown)               │
│     ├─ Order Manager (Smart Routing)                        │
│     └─ Broker Integration (Mock, Alpaca, IBKR)              │
│                                                              │
│  🔍 Validation Layer                                         │
│     ├─ Backtesting Engine                                   │
│     ├─ Walk-Forward Optimization                            │
│     └─ Monte Carlo Simulation                               │
│                                                              │
│  📈 Monitoring Layer                                         │
│     ├─ Real-time Dashboard                                  │
│     ├─ Alert Engine (Email, Slack)                          │
│     └─ Health Checker                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### ✅ 11-Phase Complete Implementation
1. **Core Infrastructure** - Config, Logging, Exceptions
2. **Resilient Data Pipeline** - Circuit breakers, caching, multi-source
3. **Feature Engineering** - Technical & statistical indicators
4. **Strategy Lifecycle** - Genetic algorithm generation
5. **RL Engine** - PPO agent with custom environment
6. **Alternative Data** - NLP sentiment analysis
7. **Adversarial Debate** - Council of agents for validation
8. **Execution & Risk** - Position sizing, kill switches
9. **API Layer** - REST + WebSocket endpoints
10. **Backtesting** - Full pipeline simulation with costs
11. **Monitoring** - Real-time dashboards & alerts

### 🎯 Example API Usage

```python
import requests

# Get current market regime
response = requests.get("http://localhost:8000/api/v1/data/regime/AAPL")
print(response.json())

# Run adversarial debate on a signal
signal = {
    "symbol": "AAPL",
    "action": "BUY",
    "confidence": 0.85,
    "price": 175.50
}
response = requests.post("http://localhost:8000/api/v1/debate/run", json=signal)
verdict = response.json()
print(f"Decision: {verdict['decision']}")
print(f"Consensus: {verdict['consensus_score']}")

# Place an order (paper trading)
order = {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "type": "market"
}
response = requests.post("http://localhost:8000/api/v1/execution/order", json=order)
print(response.json())
```

## Directory Structure

```
titan_3.0/
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
├── config/
│   └── master_config.yaml  # System configuration
├── core/                   # Core infrastructure
├── data/                   # Data ingestion & processing
├── ml/                     # Machine learning modules
├── strategies/             # Strategy definitions
├── debate/                 # Adversarial debate system
├── execution/              # Order management & risk
├── backtest/               # Backtesting engine
├── api/                    # REST API layer
├── monitoring/             # Dashboards & alerts
├── frontend/               # UI War Room
├── logs/                   # Application logs
├── models/                 # Trained ML models
├── reports/                # Backtest reports
└── data/                   # Cached market data
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the project root
cd /workspace/titan_3.0
pip install -e .
```

**API Key Errors:**
```bash
# Verify environment variables are set
echo $ALPHA_VANTAGE_KEY
```

**Port Already in Use:**
```bash
# Change ports in config/master_config.yaml
api:
  port: 8000  # Change to 8002 if 8000 is busy
monitoring:
  dashboard_port: 8001  # Change to 8003 if 8001 is busy
```

## Next Steps

1. **Customize Strategies**: Edit `strategies/` to add your own algorithms
2. **Connect Live Broker**: Update `execution/broker_interface.py` with real credentials
3. **Deploy to Cloud**: Use Docker containerization for production deployment
4. **Add More Data Sources**: Extend `data/ingestion/` with additional providers

## Support & Documentation

- Full API Docs: http://localhost:8000/docs (when running)
- Architecture Details: See `IMPLEMENTATION_PLAN.md`
- Audit Report: See `AUDIT_COMPLETE.md`

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Trading involves substantial risk. Always test thoroughly in simulation before live deployment.
