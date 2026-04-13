"""
Métricas de desempenho de estratégia de trading.

Métricas implementadas
----------------------
- Retorno total
- Sharpe Ratio anualizado
- Sortino Ratio anualizado
- Máximo Drawdown
- Win Rate
- Payoff Ratio (ganho médio / perda média)
- Profit Factor (soma ganhos / |soma perdas|)
- Número de trades
"""

import numpy as np
import pandas as pd
from typing import Optional


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Sharpe anualizado.

    periods_per_year: 8760 para 1h, 365 para 1d, 52 para 1w.
    """
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Sortino anualizado — penaliza apenas a volatilidade negativa."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Máximo Drawdown: queda máxima do pico ao vale."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def win_rate(pnl: pd.Series) -> float:
    """Porcentagem de trades com PnL positivo."""
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).mean())


def payoff_ratio(pnl: pd.Series) -> float:
    """Ganho médio / |Perda média|."""
    gains = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(gains) == 0 or len(losses) == 0:
        return 0.0
    return float(gains.mean() / abs(losses.mean()))


def profit_factor(pnl: pd.Series) -> float:
    """Soma dos ganhos / |Soma das perdas|."""
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_all(
    pnl: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    periods_per_year: int = 8760,
) -> dict:
    """
    Calcula todas as métricas e retorna dicionário.

    Parâmetros
    ----------
    pnl : pd.Series
        PnL de cada trade (em USDT ou % do equity).
    equity_curve : pd.Series, opcional
        Curva de equity acumulada. Se None, calcula a partir do pnl.
    periods_per_year : int
        Períodos por ano (para anualizar Sharpe/Sortino).
    """
    if equity_curve is None:
        equity_curve = (1 + pnl).cumprod()

    returns = pnl  # assume pnl já é retorno percentual

    metrics = {
        "n_trades": int(len(pnl)),
        "total_return_pct": float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100),
        "sharpe": round(sharpe_ratio(returns, periods_per_year), 3),
        "sortino": round(sortino_ratio(returns, periods_per_year), 3),
        "max_drawdown_pct": round(max_drawdown(equity_curve) * 100, 2),
        "win_rate_pct": round(win_rate(pnl) * 100, 1),
        "payoff_ratio": round(payoff_ratio(pnl), 3),
        "profit_factor": round(profit_factor(pnl), 3),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*50}")
    print("MÉTRICAS DE DESEMPENHO")
    print(f"{'='*50}")
    print(f"Trades           : {metrics['n_trades']}")
    print(f"Retorno Total    : {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe           : {metrics['sharpe']:.3f}")
    print(f"Sortino          : {metrics['sortino']:.3f}")
    print(f"Max Drawdown     : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate         : {metrics['win_rate_pct']:.1f}%")
    print(f"Payoff Ratio     : {metrics['payoff_ratio']:.3f}")
    print(f"Profit Factor    : {metrics['profit_factor']:.3f}")
    print(f"{'='*50}\n")
