"""
Engenharia de features ESTACIONÁRIAS.

REGRA OBRIGATÓRIA (Lopez de Prado, cap. 17-18):
  Nunca alimente o modelo com preços absolutos — eles são não-estacionários.

Features implementadas
----------------------
Grupo 1 — Momentum / Tendência
  log_ret          : retorno log de 1 período
  log_ret_5        : retorno log de 5 períodos
  ema_ratio_9_21   : (EMA_9 - EMA_21) / EMA_21
  ema_ratio_21_50  : (EMA_21 - EMA_50) / EMA_50

Grupo 2 — Volatilidade
  atr_ratio        : ATR(14) / close
  realized_vol_10  : desvio padrão de retornos log nos últimos 10 períodos

Grupo 3 — Volume / Microestrutura
  vol_ratio        : volume / SMA_20(volume)
  vol_trend        : correlação entre volume e |retorno| nos últimos 10 períodos

Grupo 4 — Diferenciação Fracionária (preserva memória)
  fracdiff         : close com diferenciação fracionária (ordem d configurável)

Grupo 5 — Futuros (opcional, se dados disponíveis)
  funding_rate, funding_rate_ma, funding_extreme
  oi_delta, oi_price_divergence
"""

import numpy as np
import pandas as pd
from typing import Optional

from data.frac_diff import frac_diff_ffd


# ── Indicadores base ──────────────────────────────────────────────────────────

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def realized_vol(close: pd.Series, window: int = 10) -> pd.Series:
    """Volatilidade realizada: desvio padrão dos retornos log na janela."""
    return log_returns(close).rolling(window).std()


def vol_trend(volume: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    """
    Correlação entre volume e |retorno| na janela — mede se volume confirma movimento.
    Valores altos: volume acompanha preço (movimento saudável).
    """
    ret_abs = log_returns(close).abs()
    return volume.rolling(window).corr(ret_abs)


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    frac_diff_d: float = 0.4,
    futures_feat: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Gera todas as features estacionárias.

    Parâmetros
    ----------
    df : pd.DataFrame
        OHLCV com colunas open, high, low, close, volume.
    frac_diff_d : float
        Ordem de diferenciação fracionária para a feature de preço.
        0 = desativado, 0.4 = padrão recomendado.
    futures_feat : pd.DataFrame, opcional
        Features de futuros (funding rate, OI etc.) já calculadas.
        Deve ter o mesmo índice que df.

    Retorno
    -------
    pd.DataFrame com todas as features estacionárias, sem NaN.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Grupo 1 — Momentum
    feat["log_ret"] = log_returns(close)
    feat["log_ret_5"] = np.log(close / close.shift(5))
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    e50 = ema(close, 50)
    feat["ema_ratio_9_21"] = (e9 - e21) / e21
    feat["ema_ratio_21_50"] = (e21 - e50) / e50

    # Grupo 2 — Volatilidade
    feat["atr_ratio"] = atr(high, low, close, 14) / close
    feat["realized_vol_10"] = realized_vol(close, 10)

    # Grupo 3 — Volume / Microestrutura
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / vol_ma
    feat["vol_trend"] = vol_trend(volume, close, 10)

    # Grupo 4 — Diferenciação Fracionária
    if frac_diff_d > 0:
        feat["fracdiff"] = frac_diff_ffd(close, d=frac_diff_d)
        # Normaliza a série fracionária (ela tem magnitude de preço)
        fd_mean = feat["fracdiff"].rolling(50).mean()
        fd_std = feat["fracdiff"].rolling(50).std()
        feat["fracdiff"] = (feat["fracdiff"] - fd_mean) / (fd_std + 1e-8)

    # Grupo 5 — Futuros (opcional)
    if futures_feat is not None and not futures_feat.empty:
        common_idx = feat.index.intersection(futures_feat.index)
        for col in futures_feat.columns:
            feat.loc[common_idx, col] = futures_feat.loc[common_idx, col]

    # Remove NaN das janelas iniciais
    feat.dropna(inplace=True)
    return feat
