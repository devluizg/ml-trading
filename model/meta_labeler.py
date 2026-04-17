"""
Meta-Labeling — Segundo modelo que aprende com os próprios trades.

Referência: Advances in Financial Machine Learning, cap. 3 (Lopez de Prado).

Como funciona
-------------
O modelo primário (RandomForest) gera sinais: LONG / SHORT / NEUTRO.
Ele sabe identificar a DIREÇÃO, mas não sabe quando está errado.

O meta-labeler é um segundo modelo treinado nos RESULTADOS dos trades:
  - Input : probabilidades do modelo primário + condições de mercado no momento
  - Output: P(WIN | sinal primário) — probabilidade de que este sinal específico vai ganhar

Se P(WIN) < threshold → descarta o sinal mesmo que o primário diga LONG/SHORT.
Se P(WIN) >= threshold → confirma e executa.

Efeito: o sistema aprende "quando o modelo principal erra" e para de repetir
os mesmos erros ao longo do tempo.

Requisito mínimo
----------------
MIN_TRADES trades resolvidos (WIN/LOSS/EXPIRED) no journal.
Abaixo disso, o meta-labeler fica inativo e o sinal primário passa direto.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

log = logging.getLogger(__name__)

MIN_TRADES      = 20      # mínimo de trades resolvidos para treinar
META_MODEL_PATH = Path("models/meta_labeler.joblib")
WIN_THRESHOLD   = 0.55    # P(WIN) mínimo para confirmar o sinal


class MetaLabeler:
    """
    Segundo modelo que filtra sinais do modelo primário.

    Treina em: probabilidades primárias + snapshot de features de mercado
    Prediz  : P(WIN) para cada novo sinal
    """

    def __init__(self, win_threshold: float = WIN_THRESHOLD):
        self.win_threshold = win_threshold
        self._model: Optional[GradientBoostingClassifier] = None
        self._scaler = StandardScaler()
        self._trained = False
        self._n_trades_trained = 0

    # ── Features para o meta-labeler ────────────────────────────────────────

    @staticmethod
    def build_meta_features(
        p_long: float,
        p_short: float,
        p_neutro: float,
        signal: int,
        market_snapshot: dict,
    ) -> pd.DataFrame:
        """
        Constrói o vetor de features para o meta-labeler.

        Combina as probabilidades do modelo primário com condições de mercado
        no momento do sinal — isso permite ao meta-labeler aprender que
        "LONG com ATR alto e funding negativo tende a perder".

        Parâmetros
        ----------
        p_long, p_short, p_neutro : float
            Probabilidades do modelo primário.
        signal : int
            Direção do sinal (1=LONG, -1=SHORT).
        market_snapshot : dict
            Features de mercado no momento do sinal (de build_features).
        """
        row = {
            "p_long":          p_long,
            "p_short":         p_short,
            "p_neutro":        p_neutro,
            "p_winner":        p_long if signal == 1 else p_short,
            "p_loser":         p_short if signal == 1 else p_long,
            "proba_gap":       abs(p_long - p_short),
            "signal_dir":      float(signal),
            # Condições de mercado
            "log_ret":         market_snapshot.get("log_ret", 0),
            "log_ret_5":       market_snapshot.get("log_ret_5", 0),
            "ema_ratio_9_21":  market_snapshot.get("ema_ratio_9_21", 0),
            "ema_ratio_21_50": market_snapshot.get("ema_ratio_21_50", 0),
            "atr_ratio":       market_snapshot.get("atr_ratio", 0),
            "realized_vol_10": market_snapshot.get("realized_vol_10", 0),
            "vol_ratio":       market_snapshot.get("vol_ratio", 0),
            "vol_trend":       market_snapshot.get("vol_trend", 0),
            "fracdiff":        market_snapshot.get("fracdiff", 0),
        }
        return pd.DataFrame([row])

    # ── Treinamento ──────────────────────────────────────────────────────────

    def fit(self, training_data: pd.DataFrame) -> bool:
        """
        Treina o meta-labeler com os trades resolvidos do journal.

        Parâmetros
        ----------
        training_data : pd.DataFrame
            Output de load_meta_training_data() — trades com outcome e features.

        Retorno
        -------
        bool — True se treinou com sucesso, False se dados insuficientes.
        """
        resolved = training_data[
            training_data["outcome"].isin(["WIN", "LOSS", "EXPIRED"])
        ].copy()

        # Só treina em LONG e SHORT (NEUTRO não tem trade real)
        resolved = resolved[resolved["signal"].isin(["LONG", "SHORT"])]

        if len(resolved) < MIN_TRADES:
            log.info(f"Meta-labeler: {len(resolved)}/{MIN_TRADES} trades resolvidos — inativo")
            return False

        # Label binário: WIN=1, LOSS/EXPIRED=0
        resolved["meta_label"] = (resolved["outcome"] == "WIN").astype(int)

        feature_cols = [
            "p_long", "p_short", "p_neutro", "p_winner", "p_loser",
            "proba_gap", "signal_dir",
            "log_ret", "log_ret_5", "ema_ratio_9_21", "ema_ratio_21_50",
            "atr_ratio", "realized_vol_10", "vol_ratio", "vol_trend", "fracdiff",
        ]
        available = [c for c in feature_cols if c in resolved.columns]
        X = resolved[available].fillna(0)
        y = resolved["meta_label"]

        if y.nunique() < 2:
            log.info("Meta-labeler: apenas uma classe nos dados — aguardando mais trades")
            return False

        X_scaled = self._scaler.fit_transform(X)
        self._model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8,
        )
        self._model.fit(X_scaled, y)
        self._trained = True
        self._n_trades_trained = len(resolved)

        win_rate = y.mean() * 100
        log.info(
            f"Meta-labeler treinado — {len(resolved)} trades | "
            f"win rate base: {win_rate:.1f}%"
        )

        # Salva em disco
        META_MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler,
                     "threshold": self.win_threshold, "n_trades": self._n_trades_trained},
                    META_MODEL_PATH)
        return True

    # ── Predição ─────────────────────────────────────────────────────────────

    def predict_win_proba(self, meta_features: pd.DataFrame) -> float:
        """Retorna P(WIN) para um conjunto de features."""
        if not self._trained or self._model is None:
            return 1.0  # inativo → deixa passar tudo

        available = [c for c in meta_features.columns if c in self._scaler.feature_names_in_]
        X = meta_features[available].fillna(0).reindex(
            columns=list(self._scaler.feature_names_in_), fill_value=0
        )
        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)
        # Classe 1 = WIN
        win_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else 1
        return float(proba[0][win_idx])

    def should_trade(
        self,
        p_long: float,
        p_short: float,
        p_neutro: float,
        signal: int,
        market_snapshot: dict,
    ) -> tuple[bool, float]:
        """
        Decide se o sinal do modelo primário deve ser executado.

        Retorno
        -------
        (executar: bool, p_win: float)
        """
        if not self._trained:
            return True, 1.0  # sem histórico → executa tudo

        meta_feat = self.build_meta_features(p_long, p_short, p_neutro, signal, market_snapshot)
        p_win = self.predict_win_proba(meta_feat)
        execute = p_win >= self.win_threshold

        log.info(f"Meta-labeler: P(WIN)={p_win:.3f} | threshold={self.win_threshold} | {'EXECUTAR' if execute else 'FILTRADO'}")
        return execute, p_win

    @property
    def is_active(self) -> bool:
        return self._trained

    @property
    def n_trades(self) -> int:
        return self._n_trades_trained


def load_meta_labeler() -> MetaLabeler:
    """Carrega meta-labeler do disco, ou retorna um novo (inativo)."""
    ml = MetaLabeler()
    if META_MODEL_PATH.exists():
        try:
            saved = joblib.load(META_MODEL_PATH)
            ml._model   = saved["model"]
            ml._scaler  = saved["scaler"]
            ml.win_threshold = saved.get("threshold", WIN_THRESHOLD)
            ml._n_trades_trained = saved.get("n_trades", 0)
            ml._trained = True
            log.info(f"Meta-labeler carregado — treinado com {ml._n_trades_trained} trades")
        except Exception as e:
            log.warning(f"Erro ao carregar meta-labeler: {e}")
    return ml
