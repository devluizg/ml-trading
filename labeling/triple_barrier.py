"""
Rotulagem pelo Método da Barreira Tripla (Triple Barrier Method).

Referência: Advances in Financial Machine Learning, Marcos Lopez de Prado,
            Capítulo 3.

Lógica
------
Para cada barra t (evento de entrada):
  - Barreira Superior (Take Profit): preço >= close[t] * (1 + pt)  → label +1
  - Barreira Inferior (Stop Loss)  : preço <= close[t] * (1 - sl)  → label -1
  - Barreira Vertical (Tempo)      : n barras expiraram sem tocar   → label  0

O primeiro evento que ocorrer determina o rótulo.
"""

import numpy as np
import pandas as pd
from typing import Optional


def apply_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    pt: float = 0.02,       # Take Profit (2%)
    sl: float = 0.02,       # Stop Loss  (2%)
    vertical_bars: int = 50,
) -> pd.Series:
    """
    Aplica o método da barreira tripla e retorna uma série de rótulos.

    Parâmetros
    ----------
    close : pd.Series
        Preços de fechamento.
    high : pd.Series
        Preços máximos (para detectar toque na barreira superior intrabar).
    low : pd.Series
        Preços mínimos (para detectar toque na barreira inferior intrabar).
    pt : float
        Fator de take profit como fração do preço de entrada.
    sl : float
        Fator de stop loss como fração do preço de entrada.
    vertical_bars : int
        Número máximo de barras na janela de observação.

    Retorno
    -------
    pd.Series com rótulos {-1, 0, +1} alinhados ao índice de `close`.
    """
    labels = []
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    n = len(close_arr)

    for i in range(n):
        entry_price = close_arr[i]
        tp_level = entry_price * (1 + pt)
        sl_level = entry_price * (1 - sl)

        label = 0  # default: barreira vertical (sem tocar TP ou SL)

        end = min(i + vertical_bars, n)
        for j in range(i + 1, end):
            if high_arr[j] >= tp_level:
                label = 1
                break
            if low_arr[j] <= sl_level:
                label = -1
                break

        labels.append(label)

    return pd.Series(labels, index=close.index, name="label")


def apply_dynamic_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    atr_multiplier_pt: float = 2.0,
    atr_multiplier_sl: float = 1.0,
    vertical_bars: int = 50,
) -> pd.Series:
    """
    Variante com barreiras dinâmicas baseadas no ATR (mais robusto).

    pt = atr[i] * atr_multiplier_pt  (Take Profit adaptativo)
    sl = atr[i] * atr_multiplier_sl  (Stop Loss adaptativo)
    """
    labels = []
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    atr_arr = atr.values
    n = len(close_arr)

    for i in range(n):
        entry_price = close_arr[i]
        atr_val = atr_arr[i]
        tp_level = entry_price + atr_multiplier_pt * atr_val
        sl_level = entry_price - atr_multiplier_sl * atr_val

        label = 0
        end = min(i + vertical_bars, n)
        for j in range(i + 1, end):
            if high_arr[j] >= tp_level:
                label = 1
                break
            if low_arr[j] <= sl_level:
                label = -1
                break

        labels.append(label)

    return pd.Series(labels, index=close.index, name="label")
