#!/bin/bash
# deploy.sh — configura a VM e instala o ml_trade bot
# Rode como root ou com sudo na máquina virtual
# Uso: bash deploy.sh

set -e

PROJECT_DIR="/home/ubuntu/ml_trade"
SERVICE_USER="ubuntu"

echo "=== ml_trade deploy ==="

# 1. Atualiza sistema e instala Python
apt-get update -q
apt-get install -y python3 python3-pip python3-venv git

# 2. Cria ambiente virtual
python3 -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"

# 3. Instala dependências
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q

echo "Dependências instaladas."

# 4. Baixa histórico de 15m (precisa de internet)
cd "$PROJECT_DIR"
python3 data/download_history.py --timeframe 15m --days 730

# 5. Otimiza parâmetros
python3 tune.py --quick --min-sharpe 0.3

# 6. Instala e ativa o serviço systemd
cp "$PROJECT_DIR/ml_trade.service" /etc/systemd/system/ml_trade.service
# Ajusta caminhos para ubuntu
sed -i "s|User=luiz|User=$SERVICE_USER|g" /etc/systemd/system/ml_trade.service
sed -i "s|/home/luiz/ml_trade|$PROJECT_DIR|g" /etc/systemd/system/ml_trade.service

systemctl daemon-reload
systemctl enable ml_trade
systemctl start ml_trade

echo ""
echo "=== Deploy concluído ==="
systemctl status ml_trade --no-pager | head -8
echo ""
echo "Logs: journalctl -u ml_trade -f"
echo "      tail -f $PROJECT_DIR/logs/bot.log"
