#!/bin/bash
# entrypoint.sh — inicia os 4 bots no container
set -e

echo "=== ml_trade — 4 bots ==="
echo "BTC 15m | BTC 1m | ETH 5m | ETH 1m"
echo ""

# Baixa históricos em paralelo (só se não existirem)
download_if_needed() {
    local symbol=$1 tf=$2 file=$3
    if [ ! -f "data/history/$file" ]; then
        echo "Baixando $symbol $tf..."
        python3 data/download_history.py --symbol "$symbol" --timeframe "$tf" --days 365
    else
        echo "Histórico $symbol $tf já existe."
    fi
}

download_if_needed "BTC/USDT" "15m" "BTC_USDT_15m.parquet" &
download_if_needed "ETH/USDT" "5m"  "ETH_USDT_5m.parquet"  &
download_if_needed "BTC/USDT" "1m"  "BTC_USDT_1m.parquet"  &
download_if_needed "ETH/USDT" "1m"  "ETH_USDT_1m.parquet"  &
wait

echo ""
echo "Iniciando bots..."

python3 main.py --loop --config config.yaml         &
sleep 5
python3 main.py --loop --config config_btc_1m.yaml  &
sleep 5
python3 main.py --loop --config config_eth.yaml     &
sleep 5
python3 main.py --loop --config config_eth_1m.yaml  &

echo "4 bots ativos. Health check: http://localhost:8080/health"

wait -n
echo "Um bot encerrou — reiniciando container..."
exit 1
