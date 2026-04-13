"""
Alertas via Telegram.

Setup (1 vez)
-------------
1. Abra o Telegram e fale com @BotFather
2. Crie um bot com /newbot, copie o TOKEN
3. Envie uma mensagem para o seu bot
4. Acesse: https://api.telegram.org/bot<TOKEN>/getUpdates
5. Copie o chat_id do campo "id"
6. Coloque TOKEN e CHAT_ID no config.yaml:
     alerts:
       telegram_token: "123456:ABC..."
       telegram_chat_id: "987654321"

Se token ou chat_id estiverem vazios, os alertas são silenciosamente ignorados.
"""

import logging
import urllib.request
import urllib.parse
import json
from typing import Optional

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _send(token: str, chat_id: str, text: str) -> bool:
    """Envia mensagem via HTTP puro (sem dependências externas)."""
    url = _TELEGRAM_API.format(token=token)
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        log.warning(f"Telegram falhou: {e}")
        return False


class TelegramAlerter:
    """
    Envia alertas do bot para o Telegram.

    Parâmetros
    ----------
    token : str
        Token do bot Telegram.
    chat_id : str
        ID do chat de destino.
    """

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token
        self.chat_id = chat_id
        self._enabled = bool(token and chat_id)
        if not self._enabled:
            log.info("Telegram desativado (token/chat_id não configurados)")

    def _msg(self, text: str) -> None:
        if not self._enabled:
            return
        _send(self.token, self.chat_id, text)

    def signal(
        self,
        symbol: str,
        signal: int,
        entry: float,
        tp: float,
        sl: float,
        p_long: float,
        p_short: float,
        n_candles: int,
        dry_run: bool,
    ) -> None:
        """Alerta de sinal de entrada."""
        label = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ NEUTRO"}[signal]
        mode = "[DRY RUN]" if dry_run else "[LIVE]"
        text = (
            f"<b>{mode} {label} — {symbol}</b>\n"
            f"Entrada : {entry:.2f}\n"
            f"TP      : {tp:.2f}  (+{abs(tp-entry)/entry*100:.1f}%)\n"
            f"SL      : {sl:.2f}  (-{abs(sl-entry)/entry*100:.1f}%)\n"
            f"P(Long) : {p_long*100:.1f}%  |  P(Short): {p_short*100:.1f}%\n"
            f"Candles treinados: {n_candles}"
        )
        self._msg(text)

    def circuit_breaker(self, reason: str) -> None:
        """Alerta de circuit breaker ativado."""
        self._msg(f"🚨 <b>CIRCUIT BREAKER</b>\n{reason}")

    def error(self, context: str, error: str) -> None:
        """Alerta de erro inesperado."""
        self._msg(f"⚠️ <b>ERRO</b> — {context}\n<code>{error[:400]}</code>")

    def startup(self, symbol: str, timeframe: str, dry_run: bool) -> None:
        """Alerta de inicialização do bot."""
        mode = "DRY RUN" if dry_run else "🔴 LIVE"
        self._msg(f"🤖 Bot iniciado — {symbol} {timeframe} [{mode}]")

    def daily_summary(self, n_signals: int, equity: float) -> None:
        """Resumo diário."""
        self._msg(
            f"📊 <b>Resumo do dia</b>\n"
            f"Sinais emitidos: {n_signals}\n"
            f"Equity atual   : {equity:.2f} USDT"
        )
