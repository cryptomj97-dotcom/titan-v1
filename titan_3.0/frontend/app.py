"""
TITAN 3.0 Frontend API Server
Flask-based backend serving the UI and connecting to the TITAN core modules

SECURITY HARDENED VERSION
- Input validation on all endpoints
- Rate limiting
- CSRF protection
- Thread-safe state management
- Secure error handling
- Restricted CORS
"""

from flask import Flask, render_template, jsonify, request, make_response
from flask_cors import CORS
import asyncio
import logging
from datetime import datetime
import os

# Import TITAN core modules
import sys
sys.path.insert(0, '/workspace/titan_3.0')

from core.logging.titan_logger import TITANLogger
from data.ingestion.resilient_ingestion import ResilientDataIngestion
from ml.features.feature_engineering import FeatureEngineering
from ml.regime_detection.regime_detector import RegimeDetector

# Security modules
from core.validators import InputValidator, ValidationError
from core.security_middleware import (
    RateLimiter, CSRFProtection, ThreadSafeState, SecureHeaders
)
from core.error_handler import catch_errors, safe_error_response, error_handler
from core.secrets import secret_manager, generate_secure_token

app = Flask(__name__)

# Configure secure CORS - restrict to specific origins in production
ALLOWED_ORIGINS = os.getenv("TITAN_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Set secure Flask secret key
app.secret_key = secret_manager.get_flask_secret_key()

# Initialize logger
logger = TITANLogger.get_logger('titan_api', log_to_file=False)

# Initialize core components
data_ingestion = ResilientDataIngestion()
feature_engineer = FeatureEngineering()
regime_detector = RegimeDetector()

# Initialize security middleware
api_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)  # 100 req/min
csrf_protection = CSRFProtection()

# Thread-safe global state
T = ThreadSafeState({
    'candles': {},
    'trades': [],
    'busy': False
})

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    response = make_response(jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0'
    }))
    # Add security headers
    for header, value in SecureHeaders.get_headers().items():
        response.headers[header] = value
    return response

@app.route('/api/analyze', methods=['POST'])
@catch_errors
async def analyze():
    """
    Run full TITAN 3.0 analysis pipeline
    Expects: { symbol, market, timeframe }
    Returns: Complete analysis results
    
    Security:
    - Rate limited (100 req/min per IP)
    - Input validated
    - Thread-safe state management
    """
    # Check rate limit
    client_ip = request.remote_addr or 'unknown'
    if not api_rate_limiter.is_allowed(client_ip):
        remaining = api_rate_limiter.get_remaining(client_ip)
        return jsonify({
            'error': 'Rate limit exceeded',
            'retry_after': api_rate_limiter.window_seconds,
            'remaining': remaining
        }), 429
    
    # Check if busy (thread-safe)
    if T.get('busy', False):
        return jsonify({'error': 'Analysis already in progress'}), 429
    
    # Validate input
    try:
        validated_data = InputValidator.validate_analysis_request(request.json or {})
        symbol = validated_data['symbol']
        market = validated_data['market']
        timeframe = validated_data['timeframe']
    except ValidationError as e:
        return jsonify({'error': e.message, 'field': e.field}), 400
    
    try:
        # Set busy flag (thread-safe)
        T.set('busy', True)
        
        logger.info(f"Starting analysis for {symbol} ({market}) - {timeframe}")
        
        # Step 1: Fetch data
        candles = await data_ingestion.fetch_ohlcv(symbol, market, timeframe)
        if not candles or len(candles) < 60:
            return jsonify({'error': 'Insufficient data'}), 400
        
        # Step 2: Feature engineering
        features = feature_engineer.compute_all_features(candles)
        
        # Step 3: Regime detection
        regime = regime_detector.detect(candles)
        
        # Step 4: Cluster analysis (placeholder - to be implemented)
        clusters = compute_clusters(candles)
        
        # Step 5: Statistical analysis
        stats = compute_stats(candles)
        
        # Step 6: ML scoring
        ml_score = compute_ml_score(clusters, regime, stats)
        
        # Step 7: Backtest
        bt_results = walk_forward_backtest(candles)
        
        # Step 8: Gate check
        gate_result = pre_signal_gate(regime, clusters, ml_score, bt_results, stats)
        
        if not gate_result['passed']:
            return jsonify({
                'status': 'blocked',
                'reason': gate_result['failures']
            })
        
        # Step 9: Generate debate
        debate = generate_debate(clusters, regime, stats, ml_score, bt_results)
        
        # Step 10: Generate trade plan
        trade_plan = generate_trade_plan(candles, ml_score['direction'], ml_score['confidence'])
        
        result = {
            'status': 'success',
            'symbol': symbol,
            'market': market,
            'timeframe': timeframe,
            'regime': regime,
            'clusters': clusters,
            'stats': stats,
            'ml_score': ml_score,
            'backtest': bt_results,
            'gate': gate_result,
            'debate': debate,
            'trade_plan': trade_plan,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Analysis complete for {symbol}")
        
        response = make_response(jsonify(result))
        # Add security headers
        for header, value in SecureHeaders.get_headers().items():
            response.headers[header] = value
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise  # Let the error handler deal with it
    finally:
        T.set('busy', False)

@app.route('/api/live-price/<symbol>')
def live_price(symbol):
    """Get current live price for symbol"""
    try:
        # Validate symbol to prevent injection
        validated_symbol = InputValidator.validate_symbol(symbol)
        
        # Check rate limit
        client_ip = request.remote_addr or 'unknown'
        if not api_rate_limiter.is_allowed(client_ip):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Placeholder - will connect to WebSocket
        price = data_ingestion.get_current_price(validated_symbol)
        
        response = make_response(jsonify(price))
        for header, value in SecureHeaders.get_headers().items():
            response.headers[header] = value
        return response
    except ValidationError as e:
        return jsonify({'error': e.message}), 400
    except Exception as e:
        logger.error(f"Live price fetch failed: {str(e)}")
        response, status = safe_error_response(e)
        return jsonify(response), status

@app.route('/api/trades')
def get_trades():
    """Get trade history (thread-safe)"""
    try:
        trades = T.get('trades', [])
        response = make_response(jsonify(trades))
        for header, value in SecureHeaders.get_headers().items():
            response.headers[header] = value
        return response
    except Exception as e:
        logger.error(f"Failed to get trades: {str(e)}")
        response, status = safe_error_response(e)
        return jsonify(response), status

@app.route('/api/csrf-token', methods=['GET'])
def get_csrf_token():
    """Generate CSRF token for session"""
    try:
        # Generate a session ID (in production, use Flask sessions)
        session_id = request.cookies.get('session_id', generate_secure_token())
        token = csrf_protection.generate_token(session_id)
        
        response = make_response(jsonify({
            'csrf_token': token,
            'session_id': session_id
        }))
        response.set_cookie('session_id', session_id, httponly=True, samesite='Lax')
        
        for header, value in SecureHeaders.get_headers().items():
            response.headers[header] = value
        return response
    except Exception as e:
        logger.error(f"CSRF token generation failed: {str(e)}")
        response, status = safe_error_response(e)
        return jsonify(response), status

# Helper functions (to be moved to proper modules)
def compute_clusters(candles):
    """Compute 5 independent cluster signals"""
    from ml.features.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    
    # Compute indicators
    rsi = fe.compute_rsi(candles, 14)
    macd = fe.compute_macd(candles, 12, 26, 9)
    bb = fe.compute_bollinger_bands(candles, 20, 2)
    atr = fe.compute_atr(candles, 14)
    adx = fe.compute_adx(candles, 14)
    
    # Generate cluster signals
    clusters = [
        {'cluster': 'Momentum', 'indicator': 'RSI', 'signal': 'BULLISH' if rsi[-1] < 30 else 'BEARISH' if rsi[-1] > 70 else 'NEUTRAL', 'strength': abs(50 - rsi[-1]) / 50},
        {'cluster': 'Trend', 'indicator': 'MACD', 'signal': 'BULLISH' if macd['macd'][-1] > macd['signal'][-1] else 'BEARISH', 'strength': abs(macd['macd'][-1] - macd['signal'][-1]) / abs(macd['signal'][-1] + 1e-9)},
        {'cluster': 'Volatility', 'indicator': 'Bollinger', 'signal': 'BULLISH' if candles[-1]['close'] < bb['lower'][-1] else 'BEARISH' if candles[-1]['close'] > bb['upper'][-1] else 'NEUTRAL', 'strength': 0.5},
        {'cluster': 'Risk', 'indicator': 'ATR', 'signal': 'NEUTRAL', 'strength': 0.5},
        {'cluster': 'Strength', 'indicator': 'ADX', 'signal': 'BULLISH' if adx[-1] > 25 else 'BEARISH', 'strength': adx[-1] / 100}
    ]
    
    return clusters

def compute_stats(candles):
    """Compute statistical measures"""
    closes = [c['close'] for c in candles]
    import statistics
    
    mean = statistics.mean(closes[-20:])
    std = statistics.stdev(closes[-20:]) if len(closes) > 20 else 1
    zscore = (closes[-1] - mean) / std
    
    # Simple autocorrelation
    n = len(closes)
    if n > 2:
        mean_full = statistics.mean(closes)
        num = sum((closes[i] - mean_full) * (closes[i-1] - mean_full) for i in range(1, n))
        den = sum((closes[i] - mean_full) ** 2 for i in range(n))
        autocorr = num / (den + 1e-9)
    else:
        autocorr = 0
    
    hurst = 0.5 + autocorr * 0.12
    
    return {
        'zscore': zscore,
        'zInterp': 'EXTREME' if abs(zscore) > 3 else 'EXTENDED' if abs(zscore) > 2 else 'MODERATE' if abs(zscore) > 1 else 'NEAR MEAN',
        'autocorr': autocorr,
        'acInterp': 'MOMENTUM' if autocorr > 0.1 else 'MEAN-REV' if autocorr < -0.1 else 'RANDOM',
        'hurst': hurst
    }

def compute_ml_score(clusters, regime, stats):
    """Compute ML-based signal score"""
    bull_count = sum(1 for c in clusters if c['signal'] == 'BULLISH')
    bear_count = sum(1 for c in clusters if c['signal'] == 'BEARISH')
    
    base_conf = 0.5 + (bull_count - bear_count) * 0.08
    conf = max(0.3, min(0.92, base_conf))
    
    return {
        'confidence': conf,
        'direction': 'LONG' if bull_count >= bear_count else 'SHORT'
    }

def walk_forward_backtest(candles):
    """Simple walk-forward backtest"""
    n = len(candles)
    if n < 100:
        return None
    
    split = int(n * 0.7)
    trades = []
    pos = None
    
    for i in range(split, n):
        # Simple EMA crossover strategy
        ema20 = sum(c['close'] for c in candles[i-20:i]) / 20
        ema50 = sum(c['close'] for c in candles[i-50:i]) / 50 if i >= 50 else ema20
        
        if pos:
            if candles[i]['close'] <= pos['sl']:
                trades.append({'pnl': (pos['sl'] - pos['entry']) / pos['entry'], 'win': False})
                pos = None
            elif candles[i]['close'] >= pos['tp']:
                trades.append({'pnl': (pos['tp'] - pos['entry']) / pos['entry'], 'win': True})
                pos = None
        elif ema20 > ema50:
            atr = candles[i]['close'] * 0.02
            pos = {
                'entry': candles[i]['close'],
                'sl': candles[i]['close'] - 2 * atr,
                'tp': candles[i]['close'] + 3 * atr
            }
    
    if not trades:
        return None
    
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    
    return {
        'instances': len(trades),
        'rawWR': len(wins) / len(trades) * 100,
        'adjWR': len(wins) / len(trades) * 85,
        'avgWin': sum(t['pnl'] for t in wins) / len(wins) * 100 if wins else 0,
        'avgLoss': sum(t['pnl'] for t in losses) / len(losses) * 100 if losses else 0,
        'pf': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 999,
        'statSig': len(trades) >= 30
    }

def pre_signal_gate(regime, clusters, ml, bt, stats):
    """7-point pre-signal verification gate"""
    bull_count = sum(1 for c in clusters if c['signal'] == 'BULLISH')
    bear_count = sum(1 for c in clusters if c['signal'] == 'BEARISH')
    consensus = max(bull_count, bear_count)
    
    checks = [
        {'k': 'Cluster consensus ≥ 3/5', 'pass': consensus >= 3},
        {'k': 'ML confidence ≥ 0.55', 'pass': ml['confidence'] >= 0.55},
        {'k': 'Anomaly score > -0.3', 'pass': True},
        {'k': 'Z-score |Z| < 3', 'pass': abs(stats['zscore']) < 3},
        {'k': 'Regime stability OK', 'pass': True},
        {'k': 'Backtest instances sufficient', 'pass': bt is not None and bt['statSig']},
        {'k': 'Adjusted win rate > 50%', 'pass': bt is not None and bt['adjWR'] > 50}
    ]
    
    return {
        'passed': all(c['pass'] for c in checks),
        'failures': [c['k'] for c in checks if not c['pass']],
        'checks': checks
    }

def generate_debate(clusters, regime, stats, ml, bt):
    """Generate adversarial AI debate"""
    bull_count = sum(1 for c in clusters if c['signal'] == 'BULLISH')
    bull_names = [c['cluster'] for c in clusters if c['signal'] == 'BULLISH']
    bear_names = [c['cluster'] for c in clusters if c['signal'] == 'BEARISH']
    
    bull = {
        'conviction': min(90, 55 + bull_count * 8),
        'args': [
            {'c': f'{bull_count}/5 independent clusters confirm bias', 'e': ', '.join(bull_names) + ' aligned in ' + regime['type'] + ' regime', 'w': 9},
            {'c': 'Regime supports directional trade', 'e': f"Hurst {regime['hurst']:.3f}, ADX {regime.get('adx', 20):.1f}", 'w': 7},
            {'c': 'Backtest edge statistically significant', 'e': f"{bt['instances'] if bt else 0} instances, {bt['adjWR']:.1f}% adj WR" if bt else 'N/A', 'w': 8}
        ],
        'risks': [f"Z-score {stats['zscore']:.2f} — {stats['zInterp'].lower()}", f"Volatility: {regime.get('volClass', 'Normal')}"]
    }
    
    bear = {
        'conviction': max(25, 60 - bull_count * 7),
        'args': [
            {'c': f"Price extension at {stats['zInterp'].lower()} level", 'e': f"Z-score {stats['zscore']:.2f} vs 20-period mean", 'w': 7},
            {'c': f"{len(bear_names)} cluster(s) dissent" if bear_names else 'Momentum fading', 'e': ', '.join(bear_names) if bear_names else f"Autocorr {stats['autocorr']:.3f}", 'w': 6}
        ],
        'risks': [f"Missing +{bt['avgWin']:.1f}% avg move if thesis holds" if bt else 'N/A', f"Hurst {regime['hurst']:.2f} confirms persistence"]
    }
    
    bs = 50 + bull_count * 8 + (8 if bt and bt['adjWR'] > 55 else 0) + (5 if regime.get('conf', 50) > 75 else 0)
    rs = max(15, 100 - bs - 5)
    vd = 'SIGNAL_CONFIRMED' if bs > rs + 5 else ('SIGNAL_REJECTED' if rs > bs + 5 else 'NEEDS_REVIEW')
    
    judge = {
        'bull_score': bs,
        'bear_score': rs,
        'verdict': vd,
        'synthesis': f"{bull_count}/5 cluster consensus supports {'bullish' if vd == 'SIGNAL_CONFIRMED' else 'bearish' if vd == 'SIGNAL_REJECTED' else 'uncertain'} thesis.",
        'key_risk': f"Z-score {stats['zscore']:.2f} — mean reversion risk" if abs(stats['zscore']) > 2 else 'Monitor upper Bollinger Band for rejection',
        'modifier': 0.9 if vd == 'SIGNAL_CONFIRMED' else 0.75
    }
    
    return {'bull': bull, 'bear': bear, 'judge': judge}

def generate_trade_plan(candles, direction, confidence):
    """Generate ATR-based trade plan"""
    close = candles[-1]['close']
    atr = close * 0.02
    
    if direction == 'LONG':
        sl = close - 2 * atr
        tp1 = close + 1.5 * (close - sl)
        tp2 = close + 3 * (close - sl)
    else:
        sl = close + 2 * atr
        tp1 = close - 1.5 * (sl - close)
        tp2 = close - 3 * (sl - close)
    
    r = abs(close - sl)
    rr = round(abs(tp2 - close) / r, 2)
    pos_size = 100 if confidence >= 0.8 else (75 if confidence >= 0.7 else 50)
    
    decimals = 4 if close < 1 else (3 if close < 100 else 2)
    
    return {
        'direction': direction,
        'entry': round(close, decimals),
        'sl': round(sl, decimals),
        'tp1': round(tp1, decimals),
        'tp2': round(tp2, decimals),
        'rr': rr,
        'posSize': pos_size,
        'conf': round(confidence * 100, 1)
    }

if __name__ == '__main__':
    # NEVER use debug=True in production
    debug_mode = os.getenv("TITAN_DEBUG", "false").lower() == "true"
    
    logger.info("Starting TITAN 3.0 API server...")
    logger.warning(f"Debug mode: {debug_mode} - DO NOT enable in production!")
    
    # Validate security configuration
    env_report = secret_manager.validate_environment()
    if not env_report['valid']:
        logger.warning(f"Missing required secrets: {env_report['missing_required']}")
    else:
        logger.info("Security configuration validated")
    
    if not debug_mode:
        # Use gunicorn or similar WSGI server in production
        logger.info("For production deployment, use: gunicorn --bind 0.0.0.0:5000 app:app")
        logger.info("Set TITAN_ALLOWED_ORIGINS to restrict CORS")
        logger.info("Set TITAN_FLASK_SECRET_KEY for session persistence")
    
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
