#!/bin/bash
# deploy.sh — instala o ml_trade na VM (Hetzner, GCP, etc.)
# Rode na VM como root: bash deploy.sh
# Uso: bash deploy.sh [usuario]   (padrão: root)

set -e

SERVICE_USER="${1:-root}"
PROJECT_DIR="/home/${SERVICE_USER}/ml_trade"
if [ "$SERVICE_USER" = "root" ]; then
    PROJECT_DIR="/root/ml_trade"
fi
VENV="$PROJECT_DIR/venv"
PYTHON="$VENV/bin/python3"

echo "=== ml_trade deploy ==="
echo "Usuário : $SERVICE_USER"
echo "Diretório: $PROJECT_DIR"
echo ""

# 1. Atualiza sistema e instala Python
apt-get update -q
apt-get install -y python3 python3-pip python3-venv -q

# 2. Cria ambiente virtual
python3 -m venv "$VENV"

# 3. Instala dependências
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q
echo "Dependências instaladas."

# 4. Cria diretórios necessários
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/data/history"
mkdir -p "$PROJECT_DIR/journal"

# 5. Baixa históricos em paralelo
echo ""
echo "Baixando históricos..."
cd "$PROJECT_DIR"
"$PYTHON" data/download_history.py --symbol BTC/USDT --timeframe 15m --days 730 &
"$PYTHON" data/download_history.py --symbol ETH/USDT --timeframe 5m  --days 730 &
"$PYTHON" data/download_history.py --symbol BTC/USDT --timeframe 1m  --days 365 &
"$PYTHON" data/download_history.py --symbol ETH/USDT --timeframe 1m  --days 365 &
wait
echo "Históricos baixados."

# 6. Instala os 4 serviços systemd
echo ""
echo "Instalando serviços systemd..."

install_service() {
    local src="$1" dst="$2"
    cp "$PROJECT_DIR/$src" "/etc/systemd/system/$dst"
    sed -i "s|User=luiz|User=$SERVICE_USER|g"     "/etc/systemd/system/$dst"
    sed -i "s|/home/luiz/ml_trade|$PROJECT_DIR|g" "/etc/systemd/system/$dst"
    sed -i "s|/usr/bin/python3|$PYTHON|g"         "/etc/systemd/system/$dst"
}

install_service "ml_trade.service"        "ml_trade.service"
install_service "ml_trade_eth.service"    "ml_trade_eth.service"
install_service "ml_trade_btc_1m.service" "ml_trade_btc_1m.service"
install_service "ml_trade_eth_1m.service" "ml_trade_eth_1m.service"

# 7. Habilita e inicia os serviços
systemctl daemon-reload

for svc in ml_trade ml_trade_eth ml_trade_btc_1m ml_trade_eth_1m; do
    systemctl enable "$svc"
done

systemctl start ml_trade
sleep 5
systemctl start ml_trade_eth
sleep 5
systemctl start ml_trade_btc_1m
sleep 5
systemctl start ml_trade_eth_1m

echo ""
echo "=== Deploy concluído — 4 bots ativos ==="
echo ""
for svc in ml_trade ml_trade_eth ml_trade_btc_1m ml_trade_eth_1m; do
    systemctl status "$svc" --no-pager | head -3
    echo ""
done
echo "Logs:"
echo "  tail -f $PROJECT_DIR/logs/bot.log"
echo "  tail -f $PROJECT_DIR/logs/bot_eth.log"
echo "  tail -f $PROJECT_DIR/logs/bot_btc_1m.log"
echo "  tail -f $PROJECT_DIR/logs/bot_eth_1m.log"
