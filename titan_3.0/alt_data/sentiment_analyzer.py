"""
TITAN 3.0 - Phase 6: Alternative Data Pipeline
Modules:
- sentiment_analyzer.py: News/Social media sentiment analysis
- macro_data.py: Macroeconomic indicator integration
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment from news headlines and social media text.
    Uses a lexicon-based approach combined with simple NLP rules.
    In production, this would integrate with HuggingFace transformers (FinBERT).
    """
    
    def __init__(self):
        # Simple financial sentiment lexicon
        self.positive_words = {
            'surge', 'soar', 'jump', 'gain', 'rally', 'bullish', 'beat', 'outperform', 
            'upgrade', 'buy', 'strong', 'growth', 'profit', 'record', 'high', 'optimistic',
            'breakout', 'momentum', 'recovery', 'upside', 'opportunity'
        }
        self.negative_words = {
            'crash', 'plunge', 'drop', 'fall', 'decline', 'bearish', 'miss', 'underperform',
            'downgrade', 'sell', 'weak', 'loss', 'low', 'pessimistic', 'breakdown',
            'slump', 'risk', 'danger', 'warning', 'crisis', 'collapse', 'downside'
        }
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Returns sentiment score (-1 to 1) and confidence.
        """
        if not text:
            return {"score": 0.0, "confidence": 0.0, "label": "NEUTRAL"}
        
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        pos_count = 0
        neg_count = 0
        negation_active = False
        
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                negation_active = True
                continue
            
            # Reset negation after 2 words
            if i > 0 and tokens[i-1] not in self.negation_words:
                negation_active = False
                
            if token in self.positive_words:
                if negation_active:
                    neg_count += 1
                else:
                    pos_count += 1
            elif token in self.negative_words:
                if negation_active:
                    pos_count += 1
                else:
                    neg_count += 1
        
        total = pos_count + neg_count
        if total == 0:
            return {"score": 0.0, "confidence": 0.0, "label": "NEUTRAL"}
        
        raw_score = (pos_count - neg_count) / total
        confidence = min(1.0, total / 5.0)  # Confidence increases with word count
        
        label = "BULLISH" if raw_score > 0.2 else ("BEARISH" if raw_score < -0.2 else "NEUTRAL")
        
        return {
            "score": round(raw_score, 3),
            "confidence": round(confidence, 3),
            "label": label,
            "pos_count": pos_count,
            "neg_count": neg_count
        }

    def analyze_batch(self, headlines: List[str]) -> pd.DataFrame:
        results = []
        for headline in headlines:
            result = self.analyze_text(headline)
            result['headline'] = headline
            results.append(result)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df['timestamp'] = datetime.now()
        
        return df

class MacroDataProcessor:
    """
    Processes macroeconomic indicators (CPI, Interest Rates, GDP, Unemployment).
    Aligns them with market data for regime correlation.
    """
    
    def __init__(self):
        self.indicators = ['CPI', 'INTEREST_RATE', 'GDP_GROWTH', 'UNEMPLOYMENT', 'PMI']
    
    def create_macro_features(self, macro_data: Dict[str, List[float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Takes raw macro data and aligns it to trading dates.
        Forward fills missing values as macro data is low frequency.
        """
        df = pd.DataFrame(index=dates)
        
        for indicator in self.indicators:
            if indicator in macro_data:
                # Create a series with the provided data points
                # Assuming macro_data[indicator] aligns with start of dates or needs interpolation
                # For simplicity, we create a dummy series and forward fill
                step = max(1, len(dates) // len(macro_data[indicator]))
                values = []
                idx = 0
                for i in range(len(dates)):
                    if idx < len(macro_data[indicator]) and i >= idx * step:
                        val = macro_data[indicator][idx]
                        idx = min(idx + 1, len(macro_data[indicator]) - 1)
                    values.append(val if 'val' in locals() else np.nan)
                
                series = pd.Series(values, index=dates)
                df[indicator] = series.ffill()
            else:
                df[indicator] = np.nan
        
        # Calculate changes (moments)
        for col in df.columns:
            df[f'{col}_CHANGE'] = df[col].diff()
            
        return df.fillna(method='bfill').fillna(0)

    def calculate_macro_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates a composite score indicating how favorable macro conditions are.
        Higher score = Favorable for risk assets.
        """
        score = pd.Series(0.0, index=df.index)
        
        # Logic: Low CPI change, Low Interest Rate, High GDP, Low Unemployment, High PMI = Good
        if 'CPI_CHANGE' in df.columns:
            score -= df['CPI_CHANGE'].clip(-0.01, 0.01) * 10  # Penalize high inflation change
            
        if 'INTEREST_RATE' in df.columns:
            score -= df['INTEREST_RATE'] * 0.5  # Penalize high rates
            
        if 'GDP_GROWTH' in df.columns:
            score += df['GDP_GROWTH'] * 2  # Reward growth
            
        if 'UNEMPLOYMENT' in df.columns:
            score -= df['UNEMPLOYMENT'] * 0.5  # Penalize unemployment
            
        if 'PMI' in df.columns:
            score += (df['PMI'] - 50) * 0.1  # Reward PMI > 50
            
        return score

class AltDataPipeline:
    """
    Main pipeline orchestrating alternative data ingestion and processing.
    """
    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.macro = MacroDataProcessor()
        
    def process_news_feed(self, headlines: List[str]) -> Dict[str, Any]:
        df = self.sentiment.analyze_batch(headlines)
        if df.empty:
            return {"avg_score": 0, "trend": "NEUTRAL"}
            
        avg_score = df['score'].mean()
        bullish_pct = len(df[df['label'] == 'BULLISH']) / len(df)
        bearish_pct = len(df[df['label'] == 'BEARISH']) / len(df)
        
        trend = "BULLISH" if bullish_pct > 0.6 else ("BEARISH" if bearish_pct > 0.6 else "NEUTRAL")
        
        return {
            "avg_score": avg_score,
            "trend": trend,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "sample_size": len(df)
        }
        
    def process_macro_indicators(self, macro_dict: Dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return self.macro.create_macro_features(macro_dict, dates)
