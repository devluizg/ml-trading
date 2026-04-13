"""
Circuit Breaker — Proteção de capital por limites de perda.

Dois níveis de proteção
-----------------------
1. Limite diário   : para o bot se perda do dia >= max_daily_loss_pct
2. Drawdown total  : para o bot se drawdown desde o pico >= max_drawdown_pct

O estado é persistido em JSON para sobreviver a reinicios do processo.

Localização: risk/circuit_state.json
"""

import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).parent / "circuit_state.json"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "peak_equity": None,
        "daily_start_equity": None,
        "daily_date": None,
        "tripped": False,
        "trip_reason": None,
    }


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def reset_circuit() -> None:
    """Reseta o circuit breaker manualmente (após revisão humana)."""
    STATE_PATH.unlink(missing_ok=True)
    log.info("Circuit breaker resetado")


class CircuitBreaker:
    """
    Monitora equity e interrompe o bot se limites forem atingidos.

    Parâmetros
    ----------
    max_daily_loss_pct : float
        Para o dia se equity_atual < equity_inicio_dia * (1 - max_daily_loss_pct).
    max_drawdown_pct : float
        Para permanentemente se equity_atual < peak_equity * (1 - max_drawdown_pct).
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self._state = _load_state()

    def update(self, equity: float) -> None:
        """
        Atualiza o estado com o equity atual.
        Deve ser chamado no início de cada ciclo do bot.
        """
        today = str(date.today())
        state = self._state

        # Inicializa pico histórico
        if state["peak_equity"] is None or equity > state["peak_equity"]:
            state["peak_equity"] = equity

        # Reinicia controle diário se novo dia
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_start_equity"] = equity
            log.info(f"Novo dia — equity início: {equity:.2f} USDT")

        _save_state(state)
        self._state = state

    def check(self, equity: float) -> tuple[bool, Optional[str]]:
        """
        Verifica se algum limite foi atingido.

        Retorno
        -------
        (pode_operar, motivo_bloqueio_ou_None)
        """
        state = self._state

        if state.get("tripped"):
            return False, state.get("trip_reason", "circuit breaker ativo")

        # Verifica drawdown máximo
        peak = state.get("peak_equity")
        if peak and peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown >= self.max_drawdown_pct:
                reason = (
                    f"Drawdown máximo atingido: {drawdown*100:.1f}% "
                    f"(limite: {self.max_drawdown_pct*100:.0f}%)"
                )
                self._trip(reason)
                return False, reason

        # Verifica perda diária
        daily_start = state.get("daily_start_equity")
        if daily_start and daily_start > 0:
            daily_loss = (daily_start - equity) / daily_start
            if daily_loss >= self.max_daily_loss_pct:
                reason = (
                    f"Limite diário atingido: -{daily_loss*100:.1f}% "
                    f"(limite: {self.max_daily_loss_pct*100:.0f}%)"
                )
                # Perda diária: para o dia mas não permanentemente
                log.warning(f"CIRCUIT BREAKER DIÁRIO: {reason}")
                return False, reason

        return True, None

    def _trip(self, reason: str) -> None:
        """Dispara o circuit breaker permanentemente (requer reset manual)."""
        self._state["tripped"] = True
        self._state["trip_reason"] = reason
        _save_state(self._state)
        log.critical(f"CIRCUIT BREAKER PERMANENTE ATIVADO: {reason}")
        log.critical("Execute reset_circuit() após revisão manual para reativar.")

    @property
    def status(self) -> dict:
        return {
            "tripped": self._state.get("tripped", False),
            "peak_equity": self._state.get("peak_equity"),
            "daily_start_equity": self._state.get("daily_start_equity"),
            "daily_date": self._state.get("daily_date"),
            "trip_reason": self._state.get("trip_reason"),
        }
