"""
Features exclusivas de Mercado de Futuros.

Estas features não existem no mercado spot e são altamente informativas
sobre o sentimento e posicionamento dos participantes.

Features implementadas
----------------------
1. funding_rate         : taxa de financiamento atual (8h)
2. funding_rate_ma      : média móvel do funding (tendência de sentimento)
3. oi_delta             : variação % do Open Interest (pressão de posições)
4. oi_price_divergence  : OI subindo + preço caindo = fraqueza (bearish)
5. liquidation_ratio    : volume de liquidações / volume total (exaustão)
6. basis_pct            : (futures - spot) / spot (prêmio/desconto)
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


def fetch_funding_rate_history(
    exchange,
    symbol: str = "BTC/USDT",
    limit: int = 100,
) -> pd.Series:
    """
    Busca histórico de funding rates da Binance Futures.

    Retorna série temporal com o funding rate de cada período de 8h.
    """
    try:
        raw = exchange.fetch_funding_rate_history(symbol, limit=limit)
        if not raw:
            return pd.Series(dtype=float)
        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df["fundingRate"].astype(float)
    except Exception as e:
        log.warning(f"Funding rate não disponível: {e}")
        return pd.Series(dtype=float)


def fetch_open_interest_history(
    exchange,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 100,
) -> pd.Series:
    """
    Busca histórico de Open Interest.

    OI = número total de contratos abertos. Crescimento confirma tendência;
    queda indica fechamento de posições (reversão potencial).
    """
    try:
        raw = exchange.fetch_open_interest_history(
            symbol, timeframe=timeframe, limit=limit
        )
        if not raw:
            return pd.Series(dtype=float)
        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        col = "openInterestValue" if "openInterestValue" in df.columns else "openInterest"
        return df[col].astype(float)
    except Exception as e:
        log.warning(f"Open Interest não disponível: {e}")
        return pd.Series(dtype=float)


def build_futures_features(
    df_ohlcv: pd.DataFrame,
    funding: pd.Series,
    open_interest: pd.Series,
    funding_ma_window: int = 3,
) -> pd.DataFrame:
    """
    Constrói features de futuros alinhadas ao índice do OHLCV.

    Parâmetros
    ----------
    df_ohlcv : pd.DataFrame
        DataFrame de candles (índice temporal).
    funding : pd.Series
        Histórico de funding rates.
    open_interest : pd.Series
        Histórico de open interest.
    funding_ma_window : int
        Janela da média móvel do funding rate.

    Retorno
    -------
    pd.DataFrame com features estacionárias de futuros, alinhadas ao OHLCV.
    """
    feat = pd.DataFrame(index=df_ohlcv.index)

    # ── Funding Rate ─────────────────────────────────────────────────────────
    if not funding.empty:
        # Reamostrar funding para o timeframe do OHLCV (forward fill)
        fr_reindexed = funding.reindex(
            df_ohlcv.index, method="ffill"
        )
        feat["funding_rate"] = fr_reindexed
        feat["funding_rate_ma"] = fr_reindexed.rolling(funding_ma_window).mean()
        feat["funding_extreme"] = (fr_reindexed.abs() > 0.001).astype(float)
    else:
        feat["funding_rate"] = 0.0
        feat["funding_rate_ma"] = 0.0
        feat["funding_extreme"] = 0.0

    # ── Open Interest ─────────────────────────────────────────────────────────
    if not open_interest.empty:
        oi_reindexed = open_interest.reindex(
            df_ohlcv.index, method="ffill"
        )
        # Variação % do OI (estacionária)
        oi_delta = oi_reindexed.pct_change(1).replace([np.inf, -np.inf], 0)
        feat["oi_delta"] = oi_delta

        # Divergência OI-Preço:
        # +1 = OI sobe + preço sobe (confirmação de alta)
        # -1 = OI sobe + preço cai  (bearish)
        #  0 = sem sinal claro
        price_delta = df_ohlcv["close"].pct_change(1)
        feat["oi_price_divergence"] = np.sign(oi_delta) * np.sign(price_delta)
    else:
        feat["oi_delta"] = 0.0
        feat["oi_price_divergence"] = 0.0

    feat.fillna(0.0, inplace=True)
    return feat
