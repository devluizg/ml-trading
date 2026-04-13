"""
Modelo de classificação multiclasse: RandomForest (padrão) ou SVM.

Inclui:
  - Treinamento com cross-validation temporal (sem data-leakage).
  - Regras de entrada baseadas em probabilidade (Lopez de Prado, cap. 10).
  - Sinal de saída: LONG, SHORT ou NEUTRO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Literal, Tuple


# ── Constantes de sinal ───────────────────────────────────────────────────────
LONG = 1
SHORT = -1
NEUTRO = 0


class TradingClassifier:
    """
    Wrapper em torno de RandomForest ou SVM com regras de entrada
    baseadas em probabilidade.

    Parâmetros
    ----------
    model_type : 'rf' | 'svm'
        Algoritmo base.
    prob_threshold : float
        Probabilidade mínima da classe vencedora para abrir posição.
    prob_gap : float
        Diferença mínima entre a melhor e segunda melhor classe.
    """

    def __init__(
        self,
        model_type: Literal["rf", "svm"] = "rf",
        prob_threshold: float = 0.45,
        prob_gap: float = 0.20,
        **model_kwargs,
    ):
        self.prob_threshold = prob_threshold
        self.prob_gap = prob_gap
        self.scaler = StandardScaler()

        if model_type == "rf":
            defaults = dict(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
            defaults.update(model_kwargs)
            self._model = RandomForestClassifier(**defaults)
        elif model_type == "svm":
            defaults = dict(kernel="rbf", probability=True, random_state=42, C=1.0)
            defaults.update(model_kwargs)
            self._model = SVC(**defaults)
        else:
            raise ValueError(f"model_type deve ser 'rf' ou 'svm', recebeu '{model_type}'")

        self.classes_ = None

    # ── Treinamento ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight=None,
    ) -> "TradingClassifier":
        """
        Treina o modelo com escalonamento dos features.

        Parâmetros
        ----------
        sample_weight : array-like, opcional
            Pesos de amostra (de model.sample_weights.get_sample_weights).
            Reduz o peso de labels sobrepostos e correlacionados.
        """
        X_scaled = self.scaler.fit_transform(X)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self._model.fit(X_scaled, y, **fit_kwargs)
        self.classes_ = self._model.classes_
        return self

    # ── Predição ──────────────────────────────────────────────────────────────

    def predict_proba_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retorna DataFrame de probabilidades com colunas nomeadas por classe."""
        X_scaled = self.scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)
        return pd.DataFrame(proba, index=X.index, columns=self.classes_)

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Aplica regras de entrada e retorna sinal {-1, 0, +1}.

        Regra LONG : P(+1) > prob_threshold  E  P(+1) - P(-1) > prob_gap
        Regra SHORT: P(-1) > prob_threshold  E  P(-1) - P(+1) > prob_gap
        Caso contrário: NEUTRO
        """
        proba = self.predict_proba_df(X)

        # Garante que as colunas existam mesmo que o modelo não as veja no treino
        for c in [-1, 0, 1]:
            if c not in proba.columns:
                proba[c] = 0.0

        p_long = proba[1]
        p_short = proba[-1]

        signals = []
        for pl, ps in zip(p_long, p_short):
            if pl > self.prob_threshold and (pl - ps) > self.prob_gap:
                signals.append(LONG)
            elif ps > self.prob_threshold and (ps - pl) > self.prob_gap:
                signals.append(SHORT)
            else:
                signals.append(NEUTRO)

        return pd.Series(signals, index=X.index, name="signal")

    # ── Avaliação ─────────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Retorna métricas de avaliação."""
        X_scaled = self.scaler.transform(X)
        y_pred = self._model.predict(X_scaled)
        label_map = {-1: "BAIXA", 0: "LATERAL", 1: "ALTA"}
        present = sorted(set(y) | set(y_pred))
        target_names = [label_map.get(c, str(c)) for c in present]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "report": classification_report(y, y_pred, labels=present, target_names=target_names),
        }
