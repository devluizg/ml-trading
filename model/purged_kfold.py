"""
PurgedKFold com Embargo — Validação cruzada para séries temporais com labels sobrepostos.

Referência: Advances in Financial Machine Learning, cap. 7 (Lopez de Prado).

Dois modos
----------
walk_forward=True  (padrão): janela expansível — treino sempre antes do teste.
  Fold 1: treina em 0..20%,    testa em 20%..40%
  Fold 2: treina em 0..40%,    testa em 40%..60%
  Fold 3: treina em 0..60%,    testa em 60%..80%
  ...
  Correto para séries temporais. Sem data leakage estrutural.

walk_forward=False: k-fold padrão com purging — permite treino antes E depois do teste.
  Útil para otimização de hiperparâmetros onde o tamanho importa mais.

Purging  : remove do treino obs cujos labels se sobrepõem com o período de teste.
Embargo  : adiciona gap entre treino e teste para evitar leakage serial.
"""

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        walk_forward: bool = True,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.walk_forward = walk_forward

    def split(self, X: pd.DataFrame, y=None, pred_times: pd.Series = None):
        indices = np.arange(len(X))
        embargo_size = int(len(X) * self.embargo_pct)
        step = len(X) // (self.n_splits + 1)

        for fold in range(self.n_splits):
            if self.walk_forward:
                # Janela expansível: teste avança, treino cresce
                test_start = (fold + 1) * step
                test_end = min(test_start + step, len(X))
                train_end = test_start - embargo_size
                if train_end < 50:
                    continue
                train_idx = indices[:train_end]
                test_idx = indices[test_start:test_end]
            else:
                # K-fold com purging (ambos os lados)
                test_start = fold * (len(X) // self.n_splits)
                test_end = test_start + (len(X) // self.n_splits)
                embargo_end = min(test_end + embargo_size, len(X))
                train_idx = np.concatenate([
                    indices[:test_start],
                    indices[embargo_end:]
                ])
                test_idx = indices[test_start:test_end]

            if pred_times is not None and len(train_idx) > 0:
                train_idx = self._purge(train_idx, test_idx, X.index, pred_times)

            if len(train_idx) < 50:
                continue

            yield train_idx, test_idx

    def _purge(self, train_idx, test_idx, index, pred_times):
        test_start_time = index[test_idx[0]]
        train_times = pred_times.iloc[train_idx]
        keep = train_times[train_times < test_start_time].index
        keep_pos = np.where(index.isin(keep))[0]
        return keep_pos


def get_pred_times(close: pd.Series, labels: pd.Series, vertical_bars: int = 50) -> pd.Series:
    n = len(close)
    t1 = []
    for i in range(n):
        end_pos = min(i + vertical_bars, n - 1)
        t1.append(close.index[end_pos])
    return pd.Series(t1, index=close.index, name="t1")
