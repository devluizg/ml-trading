"""
Conexão com a Binance Futures Testnet via ccxt.

Testnet: https://testnet.binancefuture.com
  - Crie uma conta em https://testnet.binancefuture.com
  - Gere API_KEY e API_SECRET na interface da testnet
  - Configure as variáveis de ambiente abaixo (nunca coloque as chaves no código)

Variáveis de ambiente esperadas:
  BINANCE_TESTNET_API_KEY
  BINANCE_TESTNET_API_SECRET
"""

import os
import ccxt
import pandas as pd
from typing import Optional


def get_exchange() -> ccxt.binanceusdm:
    """
    Retorna instância ccxt configurada para a Binance Futures Testnet.
    Lê credenciais de variáveis de ambiente.
    """
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")

    exchange = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "future"},
        }
    )
    exchange.set_sandbox_mode(True)  # ← aponta para testnet automaticamente
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Busca candles OHLCV da exchange e retorna DataFrame.

    Parâmetros
    ----------
    symbol    : par de negociação, ex: 'BTC/USDT'
    timeframe : '1m', '5m', '15m', '1h', '4h', '1d'
    limit     : número de candles (máx 1500 na Binance)
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


def place_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,           # 'buy' ou 'sell'
    amount: float,       # quantidade em contratos
    order_type: str = "market",
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
) -> dict:
    """
    Envia ordem para a testnet.

    Atenção: esta função só é chamada pelo main.py após validação do sinal.
    """
    order = exchange.create_order(
        symbol=symbol,
        type=order_type,
        side=side,
        amount=amount,
    )

    # Stop Loss / Take Profit via ordens separadas (Binance Futures)
    if sl_price:
        exchange.create_order(
            symbol=symbol,
            type="stop_market",
            side="sell" if side == "buy" else "buy",
            amount=amount,
            params={"stopPrice": sl_price, "reduceOnly": True},
        )

    if tp_price:
        exchange.create_order(
            symbol=symbol,
            type="take_profit_market",
            side="sell" if side == "buy" else "buy",
            amount=amount,
            params={"stopPrice": tp_price, "reduceOnly": True},
        )

    return order


def get_position(exchange: ccxt.Exchange, symbol: str) -> Optional[dict]:
    """Retorna posição aberta no símbolo, ou None se não houver."""
    positions = exchange.fetch_positions([symbol])
    for pos in positions:
        if float(pos.get("contracts", 0)) != 0:
            return pos
    return None
