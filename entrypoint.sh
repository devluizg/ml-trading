#!/bin/bash
# entrypoint.sh — inicializa os dois bots no container
set -e

echo "=== ml_trade bots ==="
echo "BTC/USDT 15m + ETH/USDT 5m | dry_run=True"
echo ""

# --- BTC/USDT 15m ---
if [ ! -f "data/history/BTC_USDT_15m.parquet" ]; then
    echo "Baixando histórico BTC/USDT 15m (2 anos)..."
    python3 data/download_history.py --symbol BTC/USDT --timeframe 15m --days 730
fi

# --- ETH/USDT 5m ---
if [ ! -f "data/history/ETH_USDT_5m.parquet" ]; then
    echo "Baixando histórico ETH/USDT 5m (2 anos)..."
    python3 data/download_history.py --symbol ETH/USDT --timeframe 5m --days 730
fi

echo ""
echo "Iniciando bot BTC..."
python3 main.py --loop &
BTC_PID=$!

echo "Iniciando bot ETH..."
python3 main.py --loop --config config_eth.yaml &
ETH_PID=$!

echo ""
echo "Bots ativos — BTC PID=$BTC_PID | ETH PID=$ETH_PID"
echo "Health check: http://localhost:8080/health"

# Fica vivo enquanto algum bot estiver rodando
wait -n $BTC_PID $ETH_PID
echo "Um bot encerrou — reiniciando container..."
exit 1
