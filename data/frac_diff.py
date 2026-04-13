"""
Diferenciação Fracionária (Fractional Differentiation).

Referência: Advances in Financial Machine Learning, cap. 5 (Lopez de Prado).

Por que não usar retorno log simples
-------------------------------------
log_ret = ln(P_t / P_{t-1}) é estacionário mas descarta toda a memória
de longo prazo da série de preços. Padrões de tendência, reversão e
sazonalidade são perdidos.

Diferenciação fracionária resolve o tradeoff
-------------------------------------------
(1-L)^d * P_t  onde d ∈ [0, 1]

- d=0   → série original (não-estacionária, com memória completa)
- d=1   → retorno simples (estacionário, sem memória)
- d=0.4 → estacionário O SUFICIENTE, mas preserva ~60% da memória

A série resultante com d ≈ 0.3-0.5 tipicamente passa no ADF test e ainda
carrega informação de médias de longo prazo, reversão à média etc.

Implementação: Fixed-Width Window (FFD)
----------------------------------------
Usa janela fixa de comprimento L para os pesos, truncada quando o peso
cai abaixo de `threshold`. Computacionalmente eficiente e produz série
sem gaps no início.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calcula os pesos da expansão binomial (1-L)^d com janela fixa.

    w_k = -w_{k-1} * (d - k + 1) / k

    Os pesos são truncados quando |w_k| < threshold.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])  # ordem: mais antigo → mais recente


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-5,
) -> pd.Series:
    """
    Aplica diferenciação fracionária com janela fixa (FFD) a uma série.

    Parâmetros
    ----------
    series : pd.Series
        Série temporal (ex: preços de fechamento).
    d : float
        Ordem de diferenciação. 0.3-0.5 é típico para preços.
    threshold : float
        Trunca pesos abaixo deste valor.

    Retorno
    -------
    pd.Series com mesma length de `series` (sem NaN iniciais — janela fixa).
    """
    weights = _get_weights_ffd(d, threshold)
    width = len(weights)
    output = []

    arr = series.values.astype(float)

    for i in range(len(arr)):
        if i < width - 1:
            # Usa pesos parciais nas primeiras observações (FFD não descarta)
            w_slice = weights[-(i + 1):]
            val = np.dot(w_slice, arr[:i + 1]) / w_slice.sum()
        else:
            val = np.dot(weights, arr[i - width + 1: i + 1])
        output.append(val)

    return pd.Series(output, index=series.index, name=f"fracdiff_d{d}")


def find_min_d(
    series: pd.Series,
    d_range: Optional[np.ndarray] = None,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
) -> float:
    """
    Encontra o menor d tal que a série diferenciada é estacionária (ADF p < adf_pvalue).

    Usa busca em grid no intervalo d_range. Retorna o menor d que passa no teste ADF.
    Se nenhum valor passar, retorna o maior d testado.

    Parâmetros
    ----------
    series : pd.Series
        Série original (preços).
    d_range : array-like, opcional
        Valores de d a testar. Padrão: [0.1, 0.2, ..., 1.0]
    adf_pvalue : float
        Nível de significância para o teste ADF.

    Retorno
    -------
    float — menor d que torna a série estacionária.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        # statsmodels não instalado: retorna d padrão conservador
        return 0.4

    if d_range is None:
        d_range = np.arange(0.1, 1.1, 0.1)

    for d in d_range:
        fd = frac_diff_ffd(series, d=d, threshold=threshold)
        result = adfuller(fd.dropna(), maxlag=1, regression="c", autolag=None)
        if result[1] <= adf_pvalue:
            return float(round(d, 2))

    return float(d_range[-1])
