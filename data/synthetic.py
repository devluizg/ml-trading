"""
Gerador de dados sintéticos para validação TDD do pipeline de ML.

Gera três regimes de mercado com labels determinísticas:
  +1 (Alta)   → tendência de alta constante
  -1 (Baixa)  → tendência de baixa constante
   0 (Lateral) → oscilação seno sem tendência líquida

A ideia: se o pipeline estiver correto, o modelo deve atingir ~100% de
acurácia nesses dados. Qualquer erro nessa etapa revela bug no código,
não no mercado.
"""

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_bars_per_regime: int = 300,
    tick_size: float = 2.0,
    amplitude_seno: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Retorna um DataFrame OHLCV com três regimes concatenados.

    Parâmetros
    ----------
    n_bars_per_regime : int
        Número de candles por regime (alta, baixa, lateral).
    tick_size : float
        Incremento/decremento por barra na tendência.
    amplitude_seno : float
        Amplitude da onda seno no regime lateral.
    seed : int
        Semente aleatória para reprodutibilidade.

    Retorno
    -------
    pd.DataFrame com colunas: open, high, low, close, volume, true_label
    """
    rng = np.random.default_rng(seed)
    frames = []

    # ── Regime ALTA (+1) ────────────────────────────────────────────────
    n = n_bars_per_regime
    close_up = 100.0 + tick_size * np.arange(n)
    noise = rng.normal(0, 0.05, n)
    close_up += noise
    high_up = close_up + rng.uniform(0.1, 0.3, n)
    low_up = close_up - rng.uniform(0.05, 0.15, n)
    open_up = np.roll(close_up, 1)
    open_up[0] = 100.0
    frames.append(
        pd.DataFrame(
            {
                "open": open_up,
                "high": high_up,
                "low": low_up,
                "close": close_up,
                "volume": rng.uniform(1000, 5000, n),
                "true_label": 1,
            }
        )
    )

    # ── Regime BAIXA (-1) ───────────────────────────────────────────────
    start_down = close_up[-1]
    close_dn = start_down - tick_size * np.arange(n)
    noise = rng.normal(0, 0.05, n)
    close_dn += noise
    high_dn = close_dn + rng.uniform(0.05, 0.15, n)
    low_dn = close_dn - rng.uniform(0.1, 0.3, n)
    open_dn = np.roll(close_dn, 1)
    open_dn[0] = start_down
    frames.append(
        pd.DataFrame(
            {
                "open": open_dn,
                "high": high_dn,
                "low": low_dn,
                "close": close_dn,
                "volume": rng.uniform(1000, 5000, n),
                "true_label": -1,
            }
        )
    )

    # ── Regime LATERAL (0) ──────────────────────────────────────────────
    start_lat = close_dn[-1]
    t = np.linspace(0, 4 * np.pi, n)
    close_lat = start_lat + amplitude_seno * np.sin(t)
    noise = rng.normal(0, 0.05, n)
    close_lat += noise
    high_lat = close_lat + rng.uniform(0.05, 0.2, n)
    low_lat = close_lat - rng.uniform(0.05, 0.2, n)
    open_lat = np.roll(close_lat, 1)
    open_lat[0] = start_lat
    frames.append(
        pd.DataFrame(
            {
                "open": open_lat,
                "high": high_lat,
                "low": low_lat,
                "close": close_lat,
                "volume": rng.uniform(1000, 5000, n),
                "true_label": 0,
            }
        )
    )

    df = pd.concat(frames, ignore_index=True)
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="1h")
    return df
