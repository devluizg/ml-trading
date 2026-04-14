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

# 5. Baixa histórico BTC 15m e ETH 5m
echo ""
echo "Baixando histórico BTC/USDT 15m..."
cd "$PROJECT_DIR"
"$PYTHON" data/download_history.py --symbol BTC/USDT --timeframe 15m --days 730

echo ""
echo "Baixando histórico ETH/USDT 5m..."
"$PYTHON" data/download_history.py --symbol ETH/USDT --timeframe 5m --days 730

# 6. Instala serviço BTC (ml_trade)
echo ""
echo "Instalando serviços systemd..."
cp "$PROJECT_DIR/ml_trade.service" /etc/systemd/system/ml_trade.service
sed -i "s|User=luiz|User=$SERVICE_USER|g"            /etc/systemd/system/ml_trade.service
sed -i "s|/home/luiz/ml_trade|$PROJECT_DIR|g"        /etc/systemd/system/ml_trade.service
sed -i "s|/usr/bin/python3|$PYTHON|g"               /etc/systemd/system/ml_trade.service

# 7. Instala serviço ETH (ml_trade_eth)
cp "$PROJECT_DIR/ml_trade_eth.service" /etc/systemd/system/ml_trade_eth.service
sed -i "s|User=luiz|User=$SERVICE_USER|g"            /etc/systemd/system/ml_trade_eth.service
sed -i "s|/home/luiz/ml_trade|$PROJECT_DIR|g"        /etc/systemd/system/ml_trade_eth.service
sed -i "s|/usr/bin/python3|$PYTHON|g"               /etc/systemd/system/ml_trade_eth.service

# 8. Habilita e inicia os serviços
systemctl daemon-reload

systemctl enable ml_trade
systemctl enable ml_trade_eth

systemctl start ml_trade
sleep 3
systemctl start ml_trade_eth

echo ""
echo "=== Deploy concluído ==="
echo ""
systemctl status ml_trade     --no-pager | head -5
echo ""
systemctl status ml_trade_eth --no-pager | head -5
echo ""
echo "Logs BTC: journalctl -u ml_trade -f"
echo "Logs ETH: journalctl -u ml_trade_eth -f"
echo "      ou: tail -f $PROJECT_DIR/logs/bot.log"
echo "      ou: tail -f $PROJECT_DIR/logs/bot_eth.log"
