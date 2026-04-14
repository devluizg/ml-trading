"""
Otimização de Hiperparâmetros — Grid Search Inteligente.

Estratégia de eficiência
------------------------
Features são computadas UMA VEZ (não dependem de pt/sl).
Labels mudam com (pt, sl, vertical_bars) — recomputadas por grupo.
Thresholds (prob_threshold, prob_gap) são aplicados no final sem re-treinar.

Isso reduz o número de backtests completos de 1200 → 60,
mas avalia todas as 1200 combinações de parâmetros.

Uso
---
  python3 tune.py                    # grid completo (~10-20 min)
  python3 tune.py --quick            # grade reduzida (~3 min)
  python3 tune.py --workers 4        # paraleliza treino por fold
  python3 tune.py --min-sharpe 0.5   # threshold para atualizar config.yaml
"""

import argparse
import copy
import itertools
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = Path("tune_results.csv")
CONFIG_PATH  = Path("config.yaml")

GRID_FULL = {
    "pt":              [0.01, 0.015, 0.02, 0.03, 0.04],
    "sl":              [0.01, 0.015, 0.02, 0.03],
    "vertical_bars":   [30, 50, 70],
    "prob_threshold":  [0.35, 0.40, 0.45, 0.50, 0.55],
    "prob_gap":        [0.10, 0.15, 0.20, 0.25],
}

GRID_QUICK = {
    "pt":              [0.015, 0.02, 0.03],
    "sl":              [0.01, 0.02],
    "vertical_bars":   [30, 50],
    "prob_threshold":  [0.40, 0.45, 0.50],
    "prob_gap":        [0.10, 0.20],
}


def evaluate_thresholds(proba_oos: pd.DataFrame, signals_grid, pnl_fn, pt, sl, vbars) -> list:
    """Avalia combinações de threshold sem re-treinar."""
    results = []
    for prob_threshold, prob_gap in signals_grid:
        # Aplica regras de entrada nas probabilidades já calculadas
        p_long  = proba_oos.get(1,  pd.Series(0, index=proba_oos.index))
        p_short = proba_oos.get(-1, pd.Series(0, index=proba_oos.index))

        signals = pd.Series(0, index=proba_oos.index, dtype=int)
        long_mask  = (p_long  > prob_threshold) & ((p_long  - p_short) > prob_gap)
        short_mask = (p_short > prob_threshold) & ((p_short - p_long)  > prob_gap)
        signals[long_mask]  = 1
        signals[short_mask] = -1

        pnl = pnl_fn(signals)
        if pnl.empty or len(pnl) < 5:
            continue

        equity = (1 + pnl).cumprod()
        from backtest.metrics import compute_all
        m = compute_all(pnl, equity)
        results.append({
            "pt": pt, "sl": sl, "vertical_bars": vbars,
            "prob_threshold": prob_threshold, "prob_gap": prob_gap,
            **m,
        })
    return results


def run_barrier_group(df, feat, config, pt, sl, vbars, threshold_combos):
    """
    Para um grupo (pt, sl, vbars): computa labels, treina walk-forward,
    coleta probabilidades OOS, depois avalia todos os thresholds.
    """
    from labeling.triple_barrier import apply_triple_barrier
    from model.classifier import TradingClassifier
    from model.purged_kfold import PurgedKFold, get_pred_times
    from model.sample_weights import get_sample_weights
    from backtest.walk_forward import simulate_trades

    mdl_cfg = config.get("model", {})

    # Labels para este grupo de barreiras
    common = feat.index.intersection(df.index)
    close = df["close"].loc[common]
    high  = df["high"].loc[common]
    low   = df["low"].loc[common]

    labels = apply_triple_barrier(close=close, high=high, low=low,
                                  pt=pt, sl=sl, vertical_bars=vbars)
    X = feat.loc[common]
    y = labels.loc[common]

    t1      = get_pred_times(close, y, vertical_bars=vbars)
    weights = get_sample_weights(close, t1)
    weights = weights.loc[X.index].fillna(1.0)

    cv = PurgedKFold(n_splits=5, embargo_pct=mdl_cfg.get("embargo_pct", 0.01), walk_forward=True)

    # Coleta probabilidades out-of-sample de todos os folds
    all_proba = []
    for train_idx, test_idx in cv.split(X, pred_times=t1):
        if len(train_idx) < 50:
            continue
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        w_tr = weights.iloc[train_idx]
        X_te = X.iloc[test_idx]

        clf = TradingClassifier(
            model_type=mdl_cfg.get("type", "rf"),
            prob_threshold=0.0, prob_gap=0.0,  # sem filtro — coleta prob brutas
            n_estimators=mdl_cfg.get("n_estimators", 200),
            max_depth=mdl_cfg.get("max_depth", 6),
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr.values)
        proba_fold = clf.predict_proba_df(X_te)
        all_proba.append(proba_fold)

    if not all_proba:
        return []

    proba_oos = pd.concat(all_proba).sort_index()

    # Função de simulação de PnL para um conjunto de sinais
    def pnl_fn(signals):
        return simulate_trades(signals, close, high, low, pt=pt, sl=sl, vertical_bars=vbars)

    return evaluate_thresholds(proba_oos, threshold_combos, pnl_fn, pt, sl, vbars)


def run_grid(grid: dict, base_config: dict, df: pd.DataFrame) -> pd.DataFrame:
    from data.features import build_features

    # Pré-computa features UMA vez
    frac_d = base_config.get("features", {}).get("frac_diff_d", 0.4)
    print("Calculando features (1 vez para todos os parâmetros)...")
    feat = build_features(df, frac_diff_d=frac_d)
    print(f"  {len(feat)} amostras, {len(feat.columns)} features")

    barrier_combos   = list(itertools.product(grid["pt"], grid["sl"], grid["vertical_bars"]))
    threshold_combos = list(itertools.product(grid["prob_threshold"], grid["prob_gap"]))
    total = len(barrier_combos)

    print(f"\n{'='*55}")
    print(f"Grid: {total} grupos de barreiras × {len(threshold_combos)} thresholds")
    print(f"= {total * len(threshold_combos)} combinações avaliadas")
    print(f"{'='*55}\n")

    all_results = []
    start = time.time()

    for i, (pt, sl, vbars) in enumerate(barrier_combos, 1):
        elapsed = time.time() - start
        rate = i / elapsed if elapsed > 0 else 0.001
        eta = (total - i) / rate
        print(f"\r  [{i}/{total}] pt={pt} sl={sl} vbars={vbars} | ETA: {eta:.0f}s | melhores: {len(all_results)}   ", end="", flush=True)

        cfg = copy.deepcopy(base_config)
        group_results = run_barrier_group(df, feat, cfg, pt, sl, vbars, threshold_combos)
        all_results.extend(group_results)

    print()
    return pd.DataFrame(all_results)


def print_top(df: pd.DataFrame, n: int = 15) -> None:
    top = df.sort_values("sharpe", ascending=False).head(n)
    print(f"\n{'='*65}")
    print(f"TOP {n} CONFIGURAÇÕES (por Sharpe)")
    print(f"{'='*65}")
    cols = ["sharpe", "sortino", "max_drawdown_pct", "win_rate_pct",
            "profit_factor", "n_trades", "pt", "sl",
            "prob_threshold", "prob_gap", "vertical_bars"]
    # Renomeia para exibição mais curta
    top_display = top[cols].copy()
    top_display.columns = ["sharpe", "sortino", "maxdd%", "wr%",
                           "pf", "trades", "pt", "sl", "thr", "gap", "vbars"]
    print(top_display.to_string(index=False, float_format="{:.3f}".format))


def apply_best(best: dict, min_sharpe: float, config_path: Path = CONFIG_PATH) -> bool:
    if best["sharpe"] < min_sharpe:
        print(f"\nMelhor Sharpe ({best['sharpe']:.3f}) < mínimo ({min_sharpe}).")
        print(f"{config_path} NÃO foi alterado.")
        print("Dica: o mercado neste período pode ser difícil para o modelo.")
        print("Considere mais dados históricos ou features adicionais (funding rate, OI).")
        return False

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["barriers"]["pt"]            = float(best["pt"])
    cfg["barriers"]["sl"]            = float(best["sl"])
    cfg["barriers"]["vertical_bars"] = int(best["vertical_bars"])
    cfg["model"]["prob_threshold"]   = float(best["prob_threshold"])
    cfg["model"]["prob_gap"]         = float(best["prob_gap"])

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ {config_path} atualizado com Sharpe={best['sharpe']:.3f}")
    print(f"  pt={best['pt']}, sl={best['sl']}, vertical_bars={int(best['vertical_bars'])}")
    print(f"  prob_threshold={best['prob_threshold']}, prob_gap={best['prob_gap']}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",      action="store_true")
    parser.add_argument("--min-sharpe", type=float, default=0.5)
    parser.add_argument("--symbol",     default=None)
    parser.add_argument("--timeframe",  default=None)
    parser.add_argument("--config",     default=None, help="Arquivo de config a atualizar (padrão: config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    results_path = Path(str(config_path).replace(".yaml", "_tune_results.csv")) if args.config else RESULTS_PATH

    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    if args.symbol:    base_config["symbol"]    = args.symbol
    if args.timeframe: base_config["timeframe"] = args.timeframe
    base_config.setdefault("features", {})["use_futures_features"] = False

    from data.storage import load_history
    df = load_history(base_config["symbol"], base_config["timeframe"])
    if df.empty:
        print("Sem dados. Rode 'python3 data/download_history.py' primeiro.")
        return

    print(f"Dados: {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Config: {config_path}")

    grid = GRID_QUICK if args.quick else GRID_FULL
    results_df = run_grid(grid, base_config, df)

    if results_df.empty:
        print("Nenhum resultado. Verifique os dados históricos.")
        return

    results_df.sort_values("sharpe", ascending=False).to_csv(results_path, index=False)
    print(f"Resultados salvos em {results_path} ({len(results_df)} combinações)")

    print_top(results_df)

    best = results_df.sort_values("sharpe", ascending=False).iloc[0].to_dict()
    apply_best(best, min_sharpe=args.min_sharpe, config_path=config_path)


if __name__ == "__main__":
    main()
