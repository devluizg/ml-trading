#!/bin/bash
# entrypoint.sh — inicializa o bot na VM/container
set -e

HISTORY_FILE="data/history/BTC_USDT_15m.parquet"

echo "=== ml_trade bot ==="
echo "Símbolo : BTC/USDT 15m"
echo "Modo    : dry_run (testnet)"
echo ""

# Baixa histórico se não existe (primeiro boot ou volume vazio)
if [ ! -f "$HISTORY_FILE" ]; then
    echo "Histórico não encontrado — baixando 2 anos de dados..."
    python3 data/download_history.py --timeframe 15m --days 730
    echo ""
    echo "Otimizando parâmetros para 15m..."
    python3 tune.py --quick --min-sharpe 0.3
    echo ""
else
    CANDLES=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$HISTORY_FILE')
print(f'{len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}')
" 2>/dev/null || echo "existente")
    echo "Histórico encontrado: $CANDLES"
fi

echo "Iniciando bot..."
exec python3 main.py --loop
