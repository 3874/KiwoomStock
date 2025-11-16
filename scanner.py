"""Market scanner that checks KOSPI and KOSDAQ stocks for overbought/oversold signals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from pykrx import stock


@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    stochastic_k_period: int = 14
    stochastic_d_period: int = 3
    bollinger_period: int = 20
    bollinger_std: int = 2
    lookback_days: int = 180


@dataclass
class StockSignal:
    code: str
    name: str
    market: str
    close: int
    rsi: float
    rsi_signal: str
    stochastic_k: float
    stochastic_d: float
    stochastic_signal: str
    bollinger_position: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "market": self.market,
            "close": self.close,
            "RSI": round(self.rsi, 2) if not np.isnan(self.rsi) else np.nan,
            "RSI 시그널": self.rsi_signal,
            "%K": round(self.stochastic_k, 2) if not np.isnan(self.stochastic_k) else np.nan,
            "%D": round(self.stochastic_d, 2) if not np.isnan(self.stochastic_d) else np.nan,
            "스토캐스틱 시그널": self.stochastic_signal,
            "볼린저 상태": self.bollinger_position,
        }


def _classify_rsi(value: float) -> str:
    if np.isnan(value):
        return "데이터 부족"
    if value >= 70:
        return "과매수"
    if value <= 30:
        return "과매도"
    return "중립"


def _classify_stochastic(k_value: float, d_value: float) -> str:
    if np.isnan(k_value) or np.isnan(d_value):
        return "데이터 부족"
    if k_value >= 80 and d_value >= 80:
        return "과매수"
    if k_value <= 20 and d_value <= 20:
        return "과매도"
    return "중립"


def _classify_bollinger(close: float, upper: float, lower: float) -> str:
    if any(np.isnan(val) for val in (close, upper, lower)):
        return "데이터 부족"
    if close > upper:
        return "상단 돌파 (과매수)"
    if close < lower:
        return "하단 돌파 (과매도)"
    return "밴드 내"


def _calculate_indicators(df: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(config.rsi_period).mean()
    avg_loss = loss.rolling(config.rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Stochastic
    lowest_low = low.rolling(config.stochastic_k_period).min()
    highest_high = high.rolling(config.stochastic_k_period).max()
    df["%K"] = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    df["%D"] = df["%K"].rolling(config.stochastic_d_period).mean()

    # Bollinger Bands
    ma = close.rolling(config.bollinger_period).mean()
    std = close.rolling(config.bollinger_period).std()
    df["UpperBand"] = ma + (std * config.bollinger_std)
    df["LowerBand"] = ma - (std * config.bollinger_std)

    return df


def _fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, code)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
        }
    )
    return df


def scan_market(market: str, config: IndicatorConfig) -> List[StockSignal]:
    end = datetime.now()
    start = end - timedelta(days=config.lookback_days)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    signals: List[StockSignal] = []

    codes = stock.get_market_ticker_list(market=market)
    for code in codes:
        df = _fetch_ohlcv(code, start_str, end_str)
        if len(df) < config.bollinger_period:
            continue
        df = _calculate_indicators(df, config)
        latest = df.iloc[-1]
        signal = StockSignal(
            code=code,
            name=stock.get_market_ticker_name(code),
            market=market,
            close=int(latest["Close"]),
            rsi=float(latest["RSI"]),
            rsi_signal=_classify_rsi(latest["RSI"]),
            stochastic_k=float(latest["%K"]),
            stochastic_d=float(latest["%D"]),
            stochastic_signal=_classify_stochastic(latest["%K"], latest["%D"]),
            bollinger_position=_classify_bollinger(
                latest["Close"], latest["UpperBand"], latest["LowerBand"]
            ),
        )
        signals.append(signal)

    return signals


def scan_markets(markets: Iterable[str] = ("KOSPI", "KOSDAQ"), config: IndicatorConfig | None = None) -> pd.DataFrame:
    config = config or IndicatorConfig()
    results: List[Dict[str, object]] = []
    for market in markets:
        market_signals = scan_market(market, config)
        results.extend(signal.as_dict() for signal in market_signals)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


__all__ = ["IndicatorConfig", "scan_markets"]
