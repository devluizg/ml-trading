"""
Orquestrador do Trading Bot — Binance Futures Testnet.

Uso
---
  python main.py                    # roda uma vez (dry-run)
  python main.py --loop             # loop contínuo no fechamento de cada candle
  python main.py --loop --live      # loop contínuo com ordens reais
  python main.py --backtest         # backtest walk-forward no histórico
  python main.py --journal          # resumo do journal de trades
  python main.py --config outro.yaml  # arquivo de configuração alternativo
"""

import argparse
import logging
import os
import sys
import time
import yaml
from datetime import datetime, timezone, date
from pathlib import Path

# ── Carrega .env antes de tudo ────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── Logging: arquivo + stdout ─────────────────────────────────────────────────
def setup_logging(log_file: str = "logs/bot.log", level: str = "INFO"):
    Path(log_file).parent.mkdir(exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    # Detecta se está rodando via systemd (evita duplicação no log file)
    under_systemd = "INVOCATION_ID" in os.environ
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if not under_systemd:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


# ── Imports do projeto ────────────────────────────────────────────────────────
import joblib
import pandas as pd

from alerts.telegram import TelegramAlerter
from backtest.walk_forward import run_walk_forward
from data.features import build_features
from data.futures_features import (
    build_futures_features,
    fetch_funding_rate_history,
    fetch_open_interest_history,
)
from data.storage import history_stats, update_history
from exchange.binance_testnet import get_exchange, get_position, place_order
from journal.trade_journal import (
    log_signal, print_summary, resolve_open_trades,
    save_feature_snapshot, load_meta_training_data,
)
from model.meta_labeler import load_meta_labeler
from labeling.triple_barrier import apply_triple_barrier, apply_dynamic_barrier
from model.classifier import LONG, NEUTRO, SHORT, TradingClassifier
from model.purged_kfold import get_pred_times
from model.sample_weights import get_sample_weights
from risk.circuit_breaker import CircuitBreaker
from risk.position_sizing import get_position_size

MODELS_DIR = Path(__file__).parent / "models"
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Persistência de modelo ────────────────────────────────────────────────────

def save_model(clf: TradingClassifier, symbol: str, timeframe: str) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    safe = symbol.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = MODELS_DIR / f"{safe}_{timeframe}_{ts}.joblib"
    joblib.dump(clf, path)
    latest = MODELS_DIR / f"{safe}_{timeframe}_latest.joblib"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


# ── Loop timer ────────────────────────────────────────────────────────────────

_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
}


def next_candle_close(timeframe: str) -> datetime:
    interval = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    now_ts = datetime.now(timezone.utc).timestamp()
    next_ts = (int(now_ts / interval) + 1) * interval + 5
    return datetime.fromtimestamp(next_ts, tz=timezone.utc)


# ── Ciclo principal ───────────────────────────────────────────────────────────

def run_bot(config: dict, dry_run: bool = True, alerter: TelegramAlerter = None):
    """Executa um ciclo completo do bot."""

    symbol = config["symbol"]
    timeframe = config["timeframe"]
    bar_cfg = config.get("barriers", {})
    mdl_cfg = config.get("model", {})
    feat_cfg = config.get("features", {})
    risk_cfg = config.get("risk", {})

    pt = bar_cfg.get("pt", 0.02)
    sl = bar_cfg.get("sl", 0.02)
    vertical_bars = bar_cfg.get("vertical_bars", 50)
    frac_diff_d = feat_cfg.get("frac_diff_d", 0.4)

    log.info(f"Ciclo iniciado — {symbol} {timeframe} | dry_run={dry_run}")

    # 1. Conexão
    exchange = get_exchange()

    # 2. Circuit breaker — verifica equity antes de operar
    circuit = CircuitBreaker(
        max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 0.03),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
    )
    try:
        balance = exchange.fetch_balance()
        equity = float(balance.get("total", {}).get("USDT", 0) or 0)
    except Exception:
        equity = 0.0

    circuit.update(equity)
    can_trade, reason = circuit.check(equity)
    if not can_trade:
        log.warning(f"Circuit breaker ativo: {reason}")
        if alerter:
            alerter.circuit_breaker(reason)
        return

    # 3. Histórico acumulado em disco
    df = update_history(
        exchange, symbol=symbol, timeframe=timeframe,
        initial_limit=config.get("initial_candles", 1000)
    )
    stats = history_stats(symbol, timeframe)
    log.info(f"Histórico: {stats['candles']} candles | {stats.get('inicio','?')} → {stats.get('fim','?')}")

    # 4. Resolve trades abertos no journal
    resolved = resolve_open_trades(df, vertical_bars=vertical_bars)
    if resolved > 0:
        log.info(f"{resolved} trade(s) resolvido(s) no journal")

    # 4b. Carrega / re-treina meta-labeler quando há novos trades resolvidos
    meta_labeler = load_meta_labeler()
    if resolved > 0 or not meta_labeler.is_active:
        training_data = load_meta_training_data()
        if not training_data.empty:
            meta_labeler.fit(training_data)
    if meta_labeler.is_active:
        log.info(f"Meta-labeler ativo — treinado com {meta_labeler.n_trades} trades")

    # 5. Features de Futuros (só na conta real — OI bloqueado no testnet)
    futures_feat = None
    if feat_cfg.get("use_futures_features", False):
        try:
            funding = fetch_funding_rate_history(exchange, symbol=symbol, limit=200)
            oi = fetch_open_interest_history(exchange, symbol=symbol, timeframe=timeframe, limit=200)
            futures_feat = build_futures_features(df, funding, oi)
        except Exception as e:
            log.warning(f"Features de futuros indisponíveis: {e}")

    # 6. Features estacionárias (incluindo frac_diff e futuros)
    feat = build_features(df, frac_diff_d=frac_diff_d, futures_feat=futures_feat)

    # 7. Barreira Tripla
    if bar_cfg.get("use_dynamic", False):
        from data.features import atr as calc_atr
        atr_series = calc_atr(df["high"], df["low"], df["close"], 14)
        labels = apply_dynamic_barrier(
            close=df["close"].loc[feat.index],
            high=df["high"].loc[feat.index],
            low=df["low"].loc[feat.index],
            atr=atr_series.loc[feat.index],
            atr_multiplier_pt=bar_cfg.get("atr_multiplier_pt", 2.0),
            atr_multiplier_sl=bar_cfg.get("atr_multiplier_sl", 1.0),
            vertical_bars=vertical_bars,
        )
    else:
        labels = apply_triple_barrier(
            close=df["close"].loc[feat.index],
            high=df["high"].loc[feat.index],
            low=df["low"].loc[feat.index],
            pt=pt, sl=sl, vertical_bars=vertical_bars,
        )

    common_idx = feat.index.intersection(labels.index)
    X = feat.loc[common_idx]
    y = labels.loc[common_idx]

    # 8. Sample weights (unicidade de labels)
    t1 = get_pred_times(df["close"].loc[common_idx], y, vertical_bars=vertical_bars)
    weights = get_sample_weights(df["close"].loc[common_idx], t1)
    weights = weights.loc[X.index].fillna(1.0)

    # Remove última barra do treino (é a barra de predição)
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    w_train = weights.iloc[:-1]
    X_current = X.iloc[[-1]]

    # 9. Treinamento com sample weights
    clf = TradingClassifier(
        model_type=mdl_cfg.get("type", "rf"),
        prob_threshold=mdl_cfg.get("prob_threshold", 0.45),
        prob_gap=mdl_cfg.get("prob_gap", 0.20),
        n_estimators=mdl_cfg.get("n_estimators", 200),
        max_depth=mdl_cfg.get("max_depth", 6),
    )
    clf.fit(X_train, y_train, sample_weight=w_train.values)

    model_path = save_model(clf, symbol, timeframe)
    log.info(f"Modelo treinado — {len(X_train)} amostras → {model_path.name}")

    # 10. Sinal atual
    signal = clf.predict_signal(X_current).iloc[0]
    proba_df = clf.predict_proba_df(X_current)
    proba = proba_df.to_dict(orient="records")[0]
    sig_label = "LONG" if signal == LONG else "SHORT" if signal == SHORT else "NEUTRO"
    log.info(f"Probabilidades: {proba}")
    log.info(f"Sinal: {sig_label}")

    current_price = float(df["close"].iloc[-1])
    sl_price = current_price * (1 - sl) if signal == LONG else current_price * (1 + sl)
    tp_price = current_price * (1 + pt) if signal == LONG else current_price * (1 - pt)

    # 10b. Meta-labeler — filtra sinais com baixa P(WIN)
    market_snapshot = X_current.iloc[0].to_dict()

    if signal != NEUTRO:
        execute, p_win = meta_labeler.should_trade(
            p_long=proba.get(1, 0),
            p_short=proba.get(-1, 0),
            p_neutro=proba.get(0, 0),
            signal=signal,
            market_snapshot=market_snapshot,
        )
        if not execute:
            log.info(
                f"Meta-labeler filtrou sinal {sig_label} — "
                f"P(WIN)={p_win:.3f} < {meta_labeler.win_threshold}"
            )
            return
    else:
        p_win = 1.0

    # 11. Journal — registra sinal; retorna timestamp para sincronizar snapshot
    signal_ts = log_signal(
        symbol=symbol, timeframe=timeframe, signal=signal, proba=proba,
        entry_price=current_price, tp_price=tp_price, sl_price=sl_price,
        dry_run=dry_run, candles_treinados=len(X_train),
    )
    if signal != NEUTRO:
        save_feature_snapshot(signal_ts, market_snapshot)

    # 12. Alerta Telegram
    if alerter and signal != NEUTRO:
        alerter.signal(
            symbol=symbol, signal=signal, entry=current_price,
            tp=tp_price, sl=sl_price,
            p_long=proba.get(1, 0), p_short=proba.get(-1, 0),
            n_candles=len(X_train), dry_run=dry_run,
        )

    if signal == NEUTRO:
        log.info("Sinal NEUTRO — nenhuma ordem enviada")
        return

    # 13. Verifica posição aberta
    current_pos = get_position(exchange, symbol)
    if current_pos:
        log.info(f"Posição já aberta — sem nova entrada")
        return

    # 14. Dimensionamento de posição
    prob_win = proba.get(1, 0) if signal == LONG else proba.get(-1, 0)
    amount = get_position_size(
        config=risk_cfg,
        equity=equity,
        entry_price=current_price,
        sl_price=sl_price,
        tp_price=tp_price,
        prob_win=prob_win,
    )

    if amount == 0:
        log.info("Kelly negativo — operação não recomendada")
        return

    # 15. Executa ordem
    side = "buy" if signal == LONG else "sell"
    if dry_run:
        log.info(
            f"[DRY RUN] Ordem simulada — side={side}, amount={amount:.3f}, "
            f"entry={current_price:.2f}, sl={sl_price:.2f}, tp={tp_price:.2f}"
        )
    else:
        try:
            order = place_order(
                exchange=exchange, symbol=symbol, side=side, amount=amount,
                sl_price=sl_price, tp_price=tp_price,
            )
            log.info(f"Ordem enviada: {order}")
        except Exception as e:
            log.error(f"Erro ao enviar ordem: {e}")
            if alerter:
                alerter.error("place_order", str(e))


# ── Loop contínuo ─────────────────────────────────────────────────────────────

def run_loop(config: dict, dry_run: bool, alerter: TelegramAlerter):
    timeframe = config["timeframe"]
    symbol = config["symbol"]
    log.info(f"Loop ativo — {symbol} {timeframe} | Ctrl+C para parar")
    if alerter:
        alerter.startup(symbol, timeframe, dry_run)

    iteration = 0
    while True:
        iteration += 1
        target = next_candle_close(timeframe)
        wait = (target - datetime.now(timezone.utc)).total_seconds()
        log.info(f"[#{iteration}] Próximo candle: {target.strftime('%H:%M:%S')} UTC — {wait:.0f}s")

        while True:
            remaining = (target - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(30, remaining))
            if (target - datetime.now(timezone.utc)).total_seconds() > 5:
                log.info(f"  ⏳ {(target - datetime.now(timezone.utc)).total_seconds():.0f}s restantes")

        log.info(f"[#{iteration}] Executando análise...")
        try:
            run_bot(config, dry_run=dry_run, alerter=alerter)
        except Exception as e:
            log.error(f"Erro na iteração #{iteration}: {e}", exc_info=True)
            if alerter:
                alerter.error(f"iteração #{iteration}", str(e))
        log.info("-" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot — Binance Futures Testnet")
    parser.add_argument("--config", default="config.yaml", help="Arquivo de configuração YAML")
    parser.add_argument("--loop", action="store_true", help="Loop contínuo no fechamento de cada candle")
    parser.add_argument("--live", action="store_true", help="Envia ordens reais (desativa dry-run)")
    parser.add_argument("--backtest", action="store_true", help="Backtest walk-forward no histórico")
    parser.add_argument("--journal", action="store_true", help="Exibe resumo do journal e sai")
    args = parser.parse_args()

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logging(log_cfg.get("file", "logs/bot.log"), log_cfg.get("level", "INFO"))

    dry_run = not args.live
    alert_cfg = config.get("alerts", {})
    alerter = TelegramAlerter(
        token=alert_cfg.get("telegram_token", ""),
        chat_id=alert_cfg.get("telegram_chat_id", ""),
    )

    if args.journal:
        print_summary()
        return

    # Gate de aprovação: bloqueia --live se backtest não passar no Sharpe mínimo
    if args.live:
        min_sharpe = config.get("live_min_sharpe", 0.5)
        log.info(f"Verificando gate de aprovação (Sharpe mínimo: {min_sharpe})...")
        from data.storage import load_history
        df_gate = load_history(config["symbol"], config["timeframe"])
        if df_gate.empty:
            log.error("Sem dados para gate de aprovação. Rode sem --live primeiro.")
            return
        gate_result = run_walk_forward(df_gate, config, n_splits=5, verbose=False)
        sharpe = gate_result.get("metrics", {}).get("sharpe", -99)
        log.info(f"Gate: Sharpe={sharpe:.3f} | Mínimo={min_sharpe}")
        if sharpe < min_sharpe:
            msg = (
                f"BLOQUEADO: Sharpe={sharpe:.3f} abaixo do mínimo={min_sharpe}.\n"
                f"Execute 'python3 tune.py' para otimizar os parâmetros."
            )
            log.critical(msg)
            if alerter:
                alerter.error("Gate de aprovação", msg)
            return
        log.info(f"Gate aprovado (Sharpe={sharpe:.3f}). Iniciando modo LIVE.")

    if args.backtest:
        log.info("Iniciando backtest walk-forward...")
        from data.storage import load_history
        df = load_history(config["symbol"], config["timeframe"])
        if df.empty:
            log.error("Sem dados históricos — rode sem --backtest primeiro para baixar")
            return
        run_walk_forward(df, config, verbose=True)
        return

    if args.loop:
        try:
            run_loop(config, dry_run=dry_run, alerter=alerter)
        except KeyboardInterrupt:
            log.info("Loop encerrado pelo usuário.")
    else:
        run_bot(config, dry_run=dry_run, alerter=alerter)


if __name__ == "__main__":
    main()
