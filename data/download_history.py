"""
Download de histórico completo da Binance Futures.

A API da Binance retorna no máximo 1500 candles por chamada.
Este script faz paginação retroativa até atingir o limite desejado
ou o início dos dados disponíveis.

Uso
---
  python3 data/download_history.py                        # BTC/USDT 1h, máximo disponível
  python3 data/download_history.py --days 365             # último ano
  python3 data/download_history.py --timeframe 4h         # timeframe diferente
  python3 data/download_history.py --symbol ETH/USDT
"""

import argparse
import sys
import time
from pathlib import Path

# Garante que raiz do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Carrega .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    import os
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import ccxt
from data.storage import save_history, load_history, HISTORY_DIR


def get_public_exchange():
    """
    Exchange pública (sem autenticação) para download de histórico.
    OHLCV é dado público — não envolve dinheiro nem credenciais.
    """
    return ccxt.binanceusdm({"options": {"defaultType": "future"}})


def download_full_history(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,       # 2 anos por padrão
    batch_size: int = 1000,
    sleep_between: float = 0.3,
) -> pd.DataFrame:
    """
    Baixa histórico completo fazendo paginação retroativa.

    Parâmetros
    ----------
    symbol : str
    timeframe : str
    days : int
        Quantos dias de histórico baixar.
    batch_size : int
        Candles por requisição (máx 1500 na Binance).
    sleep_between : float
        Pausa entre requisições para evitar rate limit.

    Retorno
    -------
    pd.DataFrame com histórico completo, mais antigo primeiro.
    """
    exchange = get_public_exchange()
    print("Fonte: Binance Futures real (dados públicos, sem autenticação)")

    # Carrega histórico existente para não rebaixar o que já temos
    existing = load_history(symbol, timeframe)
    if not existing.empty:
        oldest_existing = existing.index[0]
        print(f"Histórico existente: {len(existing)} candles ({oldest_existing} → {existing.index[-1]})")
    else:
        oldest_existing = None

    # Calcula o limite de tempo
    import time as _time
    since_ms = int((_time.time() - days * 86400) * 1000)

    all_batches = []
    until_ms = None  # None = busca do mais recente para o mais antigo

    print(f"\nBaixando {days} dias de {symbol} {timeframe}...")
    total_candles = 0

    while True:
        try:
            if until_ms is None:
                raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch_size)
            else:
                raw = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=batch_size,
                    params={"endTime": until_ms}
                )
        except Exception as e:
            print(f"\nErro na requisição: {e}")
            break

        if not raw or len(raw) == 0:
            break

        batch_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], unit="ms", utc=True)
        batch_df.set_index("timestamp", inplace=True)
        batch_df = batch_df.astype(float)

        # Remove candle atual (em formação)
        batch_df = batch_df.iloc[:-1]

        if batch_df.empty:
            break

        oldest_in_batch = batch_df.index[0]
        total_candles += len(batch_df)
        all_batches.append(batch_df)

        print(f"\r  {total_candles} candles | mais antigo: {oldest_in_batch.strftime('%Y-%m-%d')}   ", end="", flush=True)

        # Para se chegou ao limite de dias
        if oldest_in_batch.timestamp() * 1000 <= since_ms:
            break

        # Próxima paginação: busca dados ainda mais antigos
        until_ms = int(oldest_in_batch.timestamp() * 1000) - 1
        time.sleep(sleep_between)

    print()

    if not all_batches:
        print("Nenhum dado baixado.")
        return existing if not existing.empty else pd.DataFrame()

    # Concatena todos os batches e deduplicata
    new_data = pd.concat(all_batches)
    new_data = new_data[~new_data.index.duplicated(keep="first")]
    new_data.sort_index(inplace=True)

    # Filtra pelo limite de dias
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    new_data = new_data[new_data.index >= cutoff]

    # Merge com existente
    if not existing.empty:
        combined = pd.concat([new_data, existing])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_data

    save_history(combined, symbol, timeframe)
    print(f"\n✓ Histórico salvo: {len(combined)} candles")
    print(f"  Período: {combined.index[0].strftime('%Y-%m-%d')} → {combined.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Arquivo: {HISTORY_DIR}/{symbol.replace('/', '_')}_{timeframe}.parquet")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Download de histórico completo da Binance Futures")
    parser.add_argument("--symbol",    default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days",      type=int, default=730, help="Dias de histórico (padrão: 730 = 2 anos)")
    args = parser.parse_args()

    df = download_full_history(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
    )

    if not df.empty:
        print(f"\nPróximo passo: rode 'python3 tune.py --workers 4' para otimizar os parâmetros.")


if __name__ == "__main__":
    main()
