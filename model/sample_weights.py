"""
Pesos de Amostra por Unicidade de Label (Sample Uniqueness Weights).

Referência: Advances in Financial Machine Learning, cap. 4 (Lopez de Prado).

Problema que resolve
--------------------
A Barreira Tripla cria labels que se sobrepõem no tempo. Se 50 observações
consecutivas têm labels que cobrem o mesmo período, elas estão altamente
correlacionadas. Treinar com peso igual para todas infla artificialmente a
acurácia e os graus de liberdade.

Solução
-------
Calculamos a "unicidade" de cada label: quantas outras observações têm
labels que cobrem o mesmo instante de tempo? Quanto mais sobreposição,
menor o peso.

Peso_t = 1 / média(concorrência na janela de [entrada_t, saída_t])

Isso faz o modelo dar mais importância a observações únicas (com label
que cobre um período que poucos outros cobrem).
"""

import numpy as np
import pandas as pd


def get_concurrent_labels(t1: pd.Series) -> pd.Series:
    """
    Para cada instante de tempo, conta quantos labels estão "ativos"
    (i.e., quantas observações têm [entrada, saída] cobrindo esse instante).

    Parâmetros
    ----------
    t1 : pd.Series
        Index = tempo de entrada, valor = tempo de saída (fim do label).

    Retorno
    -------
    pd.Series com índice = todos os timestamps, valor = contagem de concorrência.
    """
    # Todos os timestamps únicos de início e fim
    # Normaliza timezone: garante que índice e valores sejam ambos tz-aware (UTC)
    idx = pd.DatetimeIndex(t1.index)
    vals = pd.DatetimeIndex(t1.values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    if vals.tz is None:
        vals = vals.tz_localize("UTC")
    all_times = pd.DatetimeIndex(sorted(set(idx.tolist() + vals.tolist())))
    concurrency = pd.Series(0, index=all_times, dtype=float)

    for entry, exit_ in zip(idx, vals):
        mask = (concurrency.index >= entry) & (concurrency.index <= exit_)
        concurrency[mask] += 1

    return concurrency


def get_avg_uniqueness(t1: pd.Series, concurrency: pd.Series) -> pd.Series:
    """
    Calcula a unicidade média de cada observação.

    Unicidade instantânea = 1 / concorrência no instante t.
    Unicidade média da obs = média das unicidades durante [entrada, saída].

    Parâmetros
    ----------
    t1 : pd.Series
        Index = entrada, valor = saída.
    concurrency : pd.Series
        Concorrência por timestamp (output de get_concurrent_labels).

    Retorno
    -------
    pd.Series com índice = entradas, valor = unicidade média [0, 1].
    """
    avg_uniqueness = []
    idx2 = pd.DatetimeIndex(t1.index)
    vals2 = pd.DatetimeIndex(t1.values)
    if idx2.tz is None:
        idx2 = idx2.tz_localize("UTC")
    if vals2.tz is None:
        vals2 = vals2.tz_localize("UTC")
    for entry, exit_ in zip(idx2, vals2):
        mask = (concurrency.index >= entry) & (concurrency.index <= exit_)
        c = concurrency[mask]
        if len(c) == 0 or c.sum() == 0:
            avg_uniqueness.append(1.0)
        else:
            avg_uniqueness.append((1.0 / c).mean())

    return pd.Series(avg_uniqueness, index=t1.index, name="avg_uniqueness")


def get_sample_weights(
    close: pd.Series,
    t1: pd.Series,
    return_weights: bool = True,
) -> pd.Series:
    """
    Pipeline completo: calcula pesos de amostra por unicidade.

    Parâmetros
    ----------
    close : pd.Series
        Preços de fechamento (para alinhar índice).
    t1 : pd.Series
        Index = entrada, valor = saída (de get_pred_times).
    return_weights : bool
        Se True, normaliza os pesos para somarem N (número de amostras).

    Retorno
    -------
    pd.Series com pesos para cada observação — para usar em
    model.fit(X, y, sample_weight=weights).
    """
    # Alinha t1 ao índice de close disponível
    t1_aligned = t1.loc[t1.index.isin(close.index)]
    t1_aligned = t1_aligned.clip(upper=close.index[-1])

    concurrency = get_concurrent_labels(t1_aligned)
    avg_u = get_avg_uniqueness(t1_aligned, concurrency)

    if return_weights:
        # Normaliza: soma dos pesos = número de amostras
        weights = avg_u / avg_u.mean()
    else:
        weights = avg_u

    return weights.reindex(close.index).fillna(1.0)
