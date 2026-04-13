"""
Backtesting Walk-Forward com PurgedKFold.

O que é Walk-Forward
--------------------
Em vez de treinar uma vez e testar uma vez, dividimos o histórico em N folds.
Para cada fold:
  1. Treina no passado (com PurgedKFold interno para otimização)
  2. Prediz no futuro imediato (out-of-sample)
  3. Simula os trades com gestão de risco
  4. Registra PnL

O resultado é uma equity curve que representa o que o bot TERIA feito em
dados que o modelo nunca viu — a estimativa mais honesta de desempenho.

Uso
---
    results = run_walk_forward(df, config)
    print_metrics(results["metrics"])
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtest.metrics import compute_all, print_metrics
from data.features import build_features
from labeling.triple_barrier import apply_triple_barrier
from model.classifier import TradingClassifier
from model.purged_kfold import PurgedKFold, get_pred_times
from model.sample_weights import get_sample_weights

log = logging.getLogger(__name__)


def simulate_trades(
    signals: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    pt: float = 0.02,
    sl: float = 0.02,
    vertical_bars: int = 50,
) -> pd.Series:
    """
    Simula trades barra a barra a partir dos sinais e retorna PnL por trade.

    Para cada sinal LONG ou SHORT:
    - Entra no close da barra do sinal
    - Monitora as próximas N barras para TP, SL ou expiração
    - Calcula PnL como retorno percentual

    Retorno
    -------
    pd.Series com PnL percentual de cada trade (alinhada ao índice de signals).
    """
    pnl_list = []
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    signal_arr = signals.reindex(close.index).fillna(0).values
    n = len(close_arr)

    for i in range(n):
        sig = signal_arr[i]
        if sig == 0:
            continue

        entry = close_arr[i]
        if sig == 1:  # LONG
            tp_level = entry * (1 + pt)
            sl_level = entry * (1 - sl)
        else:  # SHORT
            tp_level = entry * (1 - pt)
            sl_level = entry * (1 + sl)

        result = 0.0
        end = min(i + vertical_bars, n)
        for j in range(i + 1, end):
            if sig == 1:
                if high_arr[j] >= tp_level:
                    result = pt
                    break
                if low_arr[j] <= sl_level:
                    result = -sl
                    break
            else:
                if low_arr[j] <= tp_level:
                    result = pt
                    break
                if high_arr[j] >= sl_level:
                    result = -sl
                    break
        # Se expirou sem tocar: result = retorno do close
        if result == 0.0:
            exit_price = close_arr[min(i + vertical_bars, n - 1)]
            if sig == 1:
                result = (exit_price - entry) / entry
            else:
                result = (entry - exit_price) / entry

        pnl_list.append({"timestamp": close.index[i], "pnl": result, "signal": sig})

    if not pnl_list:
        return pd.Series(dtype=float)

    df_pnl = pd.DataFrame(pnl_list).set_index("timestamp")
    return df_pnl["pnl"]


def run_walk_forward(
    df: pd.DataFrame,
    config: dict,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Executa backtesting walk-forward completo.

    Parâmetros
    ----------
    df : pd.DataFrame
        Histórico OHLCV completo.
    config : dict
        Configuração do bot (seções 'barriers', 'model', 'features').
    n_splits : int
        Número de folds walk-forward.
    verbose : bool
        Imprime progresso de cada fold.

    Retorno
    -------
    dict com:
        "pnl"          : pd.Series com PnL de cada trade
        "equity_curve" : pd.Series com equity cumulativa
        "metrics"      : dict com Sharpe, drawdown etc.
        "fold_results" : lista com resultado de cada fold
    """
    bar_cfg = config.get("barriers", {})
    mdl_cfg = config.get("model", {})
    feat_cfg = config.get("features", {})

    pt = bar_cfg.get("pt", 0.02)
    sl = bar_cfg.get("sl", 0.02)
    vertical_bars = bar_cfg.get("vertical_bars", 50)
    prob_threshold = mdl_cfg.get("prob_threshold", 0.45)
    prob_gap = mdl_cfg.get("prob_gap", 0.20)
    frac_diff_d = feat_cfg.get("frac_diff_d", 0.4)
    embargo_pct = mdl_cfg.get("embargo_pct", 0.01)

    # Features e labels sobre o histórico completo
    feat = build_features(df, frac_diff_d=frac_diff_d)
    common = feat.index.intersection(df.index)
    close = df["close"].loc[common]
    high = df["high"].loc[common]
    low = df["low"].loc[common]

    labels = apply_triple_barrier(
        close=close, high=high, low=low, pt=pt, sl=sl, vertical_bars=vertical_bars
    )

    X = feat.loc[common]
    y = labels.loc[common]

    # t1 para PurgedKFold
    t1 = get_pred_times(close, y, vertical_bars=vertical_bars)
    t1 = t1.loc[X.index]

    # Pesos de amostra
    weights = get_sample_weights(close, t1)
    weights = weights.loc[X.index].fillna(1.0)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    all_signals = pd.Series(0, index=X.index, dtype=float)
    fold_results = []

    for fold_n, (train_idx, test_idx) in enumerate(cv.split(X, pred_times=t1), 1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        w_train = weights.iloc[train_idx]
        X_test = X.iloc[test_idx]

        if len(X_train) < 50:
            log.warning(f"Fold {fold_n}: treino muito pequeno ({len(X_train)}), pulando")
            continue

        clf = TradingClassifier(
            model_type=mdl_cfg.get("type", "rf"),
            prob_threshold=prob_threshold,
            prob_gap=prob_gap,
            n_estimators=mdl_cfg.get("n_estimators", 200),
            max_depth=mdl_cfg.get("max_depth", 6),
        )
        clf.fit(X_train, y_train, sample_weight=w_train.values)
        signals_fold = clf.predict_signal(X_test)
        all_signals.iloc[test_idx] = signals_fold.values

        n_long = (signals_fold == 1).sum()
        n_short = (signals_fold == -1).sum()
        n_neutro = (signals_fold == 0).sum()

        fold_results.append({
            "fold": fold_n,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_long": n_long,
            "n_short": n_short,
            "n_neutro": n_neutro,
        })

        if verbose:
            log.info(
                f"Fold {fold_n}/{n_splits}: treino={len(X_train)}, "
                f"teste={len(X_test)}, LONG={n_long}, SHORT={n_short}, NEUTRO={n_neutro}"
            )

    # Simula trades com os sinais OOS
    pnl = simulate_trades(
        signals=all_signals,
        close=close,
        high=high,
        low=low,
        pt=pt,
        sl=sl,
        vertical_bars=vertical_bars,
    )

    if pnl.empty:
        log.warning("Nenhum trade gerado no backtest")
        return {"pnl": pnl, "equity_curve": pd.Series(), "metrics": {}, "fold_results": fold_results}

    equity_curve = (1 + pnl).cumprod()
    metrics = compute_all(pnl, equity_curve, periods_per_year=8760)

    if verbose:
        print_metrics(metrics)

    return {
        "pnl": pnl,
        "equity_curve": equity_curve,
        "signals": all_signals,
        "metrics": metrics,
        "fold_results": fold_results,
    }
