"""
Persistência de dados históricos OHLCV.

Lógica de crescimento
---------------------
- Primeira execução: baixa os últimos N candles da API e salva em Parquet.
- Execuções seguintes: carrega do disco, baixa apenas os candles novos
  (desde o último timestamp salvo), faz append e salva novamente.
- O modelo passa a treinar em TODOS os dados acumulados, não apenas nos
  últimos 500 candles da API.

Localização dos arquivos
------------------------
  data/history/<SYMBOL>_<TIMEFRAME>.parquet
  ex: data/history/BTC_USDT_1h.parquet
"""

from pathlib import Path

import pandas as pd

HISTORY_DIR = Path(__file__).parent / "history"


def _parquet_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return HISTORY_DIR / f"{safe}_{timeframe}.parquet"


def load_history(symbol: str, timeframe: str) -> pd.DataFrame:
    """Carrega histórico do disco. Retorna DataFrame vazio se não existir."""
    path = _parquet_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception:
        # Arquivo corrompido — remove para forçar novo download
        path.unlink(missing_ok=True)
        return pd.DataFrame()


def save_history(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Salva (sobrescreve) o histórico completo em disco."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(symbol, timeframe)
    df.to_parquet(path)


def update_history(
    exchange,
    symbol: str,
    timeframe: str,
    initial_limit: int = 1000,
) -> pd.DataFrame:
    """
    Carrega histórico do disco e busca apenas os candles novos na API.

    Na primeira execução baixa `initial_limit` candles.
    Nas seguintes, baixa apenas os candles posteriores ao último salvo.

    Retorna o DataFrame completo e atualizado.
    """
    from exchange.binance_testnet import fetch_ohlcv

    existing = load_history(symbol, timeframe)

    if existing.empty:
        # Primeira execução: bootstrap com histórico inicial
        df_new = fetch_ohlcv(exchange, symbol=symbol, timeframe=timeframe, limit=initial_limit)
        save_history(df_new, symbol, timeframe)
        return df_new

    # Execuções seguintes: busca só o que falta
    last_ts = existing.index[-1]
    # Garante timezone awareness consistente para comparação
    if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is None:
        last_ts = pd.Timestamp(last_ts).tz_localize("UTC")
    df_new = fetch_ohlcv(exchange, symbol=symbol, timeframe=timeframe, limit=100)
    if df_new.index.tz is None:
        df_new.index = df_new.index.tz_localize("UTC")

    # Filtra apenas candles mais novos que o último salvo
    df_new = df_new[df_new.index > last_ts]

    if df_new.empty:
        return existing  # Nada novo ainda

    # Descarta o último candle novo se ainda não fechou
    # (a Binance retorna o candle atual em formação — removemos)
    df_new = df_new.iloc[:-1]

    if df_new.empty:
        return existing

    combined = pd.concat([existing, df_new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    save_history(combined, symbol, timeframe)
    return combined


def history_stats(symbol: str, timeframe: str) -> dict:
    """Retorna estatísticas do histórico salvo."""
    df = load_history(symbol, timeframe)
    if df.empty:
        return {"candles": 0}
    return {
        "candles": len(df),
        "inicio": str(df.index[0]),
        "fim": str(df.index[-1]),
        "arquivo": str(_parquet_path(symbol, timeframe)),
    }
