"""
Journal de trades: registra sinais e acompanha resultados.

Colunas
-------
Abertura : timestamp, symbol, timeframe, signal, p_long, p_short, p_neutro,
           entry_price, tp_price, sl_price, dry_run, candles_treinados
Resultado: outcome (WIN/LOSS/EXPIRED), exit_price, pnl_pct, resolved_at

O método `resolve_open_trades` é chamado a cada ciclo do bot. Ele varre
os trades abertos, verifica se o preço atingiu TP/SL/expirou nos candles
subsequentes e preenche as colunas de resultado.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

JOURNAL_PATH   = Path(__file__).parent / "trades.csv"
SNAPSHOTS_PATH = Path(__file__).parent / "feature_snapshots.jsonl"

_COLUMNS = [
    "timestamp",
    "symbol",
    "timeframe",
    "signal",
    "p_long",
    "p_short",
    "p_neutro",
    "entry_price",
    "tp_price",
    "sl_price",
    "dry_run",
    "candles_treinados",
    "outcome",       # WIN / LOSS / EXPIRED / OPEN
    "exit_price",
    "pnl_pct",
    "resolved_at",
]

log = logging.getLogger(__name__)


def log_signal(
    symbol: str,
    timeframe: str,
    signal: int,
    proba: dict,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    dry_run: bool,
    candles_treinados: int,
    timestamp: Optional[str] = None,
) -> str:
    """
    Registra um novo sinal no journal com outcome=OPEN.

    Retorna o timestamp usado, para que o caller possa associar
    o snapshot de features ao mesmo registro.
    """
    signal_map = {1: "LONG", -1: "SHORT", 0: "NEUTRO"}
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": ts,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal_map.get(signal, str(signal)),
        "p_long": round(proba.get(1, 0), 4),
        "p_short": round(proba.get(-1, 0), 4),
        "p_neutro": round(proba.get(0, 0), 4),
        "entry_price": entry_price,
        "tp_price": round(tp_price, 4),
        "sl_price": round(sl_price, 4),
        "dry_run": dry_run,
        "candles_treinados": candles_treinados,
        "outcome": "OPEN" if signal != 0 else "NEUTRO",
        "exit_price": "",
        "pnl_pct": "",
        "resolved_at": "",
    }
    write_header = not JOURNAL_PATH.exists()
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return ts


def load_journal() -> pd.DataFrame:
    """Carrega o journal como DataFrame."""
    if not JOURNAL_PATH.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(JOURNAL_PATH, parse_dates=["timestamp"])
    return df


def resolve_open_trades(
    df_ohlcv: pd.DataFrame,
    vertical_bars: int = 50,
) -> int:
    """
    Verifica trades ainda OPEN e preenche outcome/exit_price/pnl_pct.

    Para cada trade OPEN, olha os candles posteriores à entrada e verifica
    se TP, SL ou barreira vertical foram atingidos.

    Parâmetros
    ----------
    df_ohlcv : pd.DataFrame
        Histórico OHLCV completo (precisa cobrir o período após os trades).
    vertical_bars : int
        Janela máxima da barreira vertical.

    Retorno
    -------
    int — número de trades resolvidos nesta chamada.
    """
    journal = load_journal()
    if journal.empty:
        return 0

    # Migração: adiciona colunas ausentes em journals antigos
    for col in ["outcome", "exit_price", "pnl_pct", "resolved_at"]:
        if col not in journal.columns:
            journal[col] = ""
    if "outcome" in journal.columns:
        # Trades sem outcome preenchido = OPEN (registros do formato antigo)
        journal.loc[journal["outcome"].isna() | (journal["outcome"] == ""), "outcome"] = \
            journal.loc[journal["outcome"].isna() | (journal["outcome"] == ""), "signal"].apply(
                lambda s: "NEUTRO" if s == "NEUTRO" else "OPEN"
            )

    open_trades = journal[journal["outcome"] == "OPEN"].copy()
    if open_trades.empty:
        return 0

    resolved = 0
    updated_rows = journal.copy()

    for idx, trade in open_trades.iterrows():
        entry_time = pd.to_datetime(trade["timestamp"], utc=True)
        signal = trade["signal"]
        entry_price = float(trade["entry_price"])
        tp_price = float(trade["tp_price"])
        sl_price = float(trade["sl_price"])

        # Candles após a entrada
        future = df_ohlcv[df_ohlcv.index > entry_time].iloc[:vertical_bars]
        if future.empty:
            continue  # ainda não há dados suficientes

        outcome = "EXPIRED"
        exit_price = float(future["close"].iloc[-1])

        for _, candle in future.iterrows():
            if signal == "LONG":
                if candle["high"] >= tp_price:
                    outcome = "WIN"
                    exit_price = tp_price
                    break
                if candle["low"] <= sl_price:
                    outcome = "LOSS"
                    exit_price = sl_price
                    break
            elif signal == "SHORT":
                if candle["low"] <= tp_price:
                    outcome = "WIN"
                    exit_price = tp_price
                    break
                if candle["high"] >= sl_price:
                    outcome = "LOSS"
                    exit_price = sl_price
                    break

        if signal == "LONG":
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 3)
        elif signal == "SHORT":
            pnl_pct = round((entry_price - exit_price) / entry_price * 100, 3)
        else:
            pnl_pct = 0.0

        updated_rows.at[idx, "outcome"] = outcome
        updated_rows.at[idx, "exit_price"] = exit_price
        updated_rows.at[idx, "pnl_pct"] = pnl_pct
        updated_rows.at[idx, "resolved_at"] = datetime.now(timezone.utc).isoformat()
        resolved += 1
        log.info(f"Trade resolvido: {signal} @ {entry_price:.2f} → {outcome} ({pnl_pct:+.2f}%)")

    if resolved > 0:
        updated_rows.to_csv(JOURNAL_PATH, index=False)

    return resolved


def save_feature_snapshot(timestamp: str, features: dict) -> None:
    """
    Salva snapshot de features de mercado no momento do sinal.

    Usado pelo meta-labeler para montar o dataset de treinamento:
    combina estes features com o outcome real do trade (WIN/LOSS/EXPIRED).
    """
    record = {"timestamp": timestamp, **features}
    with open(SNAPSHOTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_meta_training_data() -> pd.DataFrame:
    """
    Combina journal (outcomes) com snapshots de features para treinar o meta-labeler.

    Retorna DataFrame com colunas:
        signal, outcome, p_long, p_short, p_neutro,
        p_winner, p_loser, proba_gap, signal_dir,
        log_ret, log_ret_5, ema_ratio_*, atr_ratio, ...
    """
    journal = load_journal()
    if journal.empty:
        return pd.DataFrame()

    if not SNAPSHOTS_PATH.exists():
        return pd.DataFrame()

    rows = []
    with open(SNAPSHOTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    snapshots = pd.DataFrame(rows)

    # Só trades com outcome definido (não OPEN, não NEUTRO)
    resolved = journal[journal["outcome"].isin(["WIN", "LOSS", "EXPIRED"])].copy()
    if resolved.empty:
        return pd.DataFrame()

    merged = resolved.merge(snapshots, on="timestamp", how="inner")

    # Colunas derivadas que o meta-labeler usa
    if "p_long" in merged.columns and "p_short" in merged.columns:
        sig_numeric = merged["signal"].map({"LONG": 1, "SHORT": -1}).fillna(0)
        merged["p_winner"]   = merged.apply(
            lambda r: r["p_long"] if r["signal"] == "LONG" else r["p_short"], axis=1
        )
        merged["p_loser"]    = merged.apply(
            lambda r: r["p_short"] if r["signal"] == "LONG" else r["p_long"], axis=1
        )
        merged["proba_gap"]  = (merged["p_long"] - merged["p_short"]).abs()
        merged["signal_dir"] = sig_numeric.astype(float)

    return merged


def print_summary() -> None:
    """Imprime resumo completo do journal no terminal."""
    df = load_journal()
    if df.empty:
        print("Journal vazio — nenhum sinal registrado ainda.")
        return

    total = len(df)
    sinais = df[df["signal"] != "NEUTRO"]
    resolvidos = sinais[sinais["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]

    print(f"\n{'='*55}")
    print(f"JOURNAL DE TRADES — {JOURNAL_PATH.name}")
    print(f"{'='*55}")
    print(f"Total de sinais  : {total}")

    for sig in ["LONG", "SHORT", "NEUTRO"]:
        n = (df["signal"] == sig).sum()
        print(f"  {sig:<8}       : {n} ({n/total*100:.1f}%)")

    if not resolvidos.empty:
        pnl = pd.to_numeric(resolvidos["pnl_pct"], errors="coerce").dropna()
        wins = (resolvidos["outcome"] == "WIN").sum()
        losses = (resolvidos["outcome"] == "LOSS").sum()
        expired = (resolvidos["outcome"] == "EXPIRED").sum()
        wr = wins / len(resolvidos) * 100 if len(resolvidos) > 0 else 0

        print(f"\nTrades resolvidos: {len(resolvidos)}")
        print(f"  WIN     : {wins}  ({wr:.1f}%)")
        print(f"  LOSS    : {losses}")
        print(f"  EXPIRED : {expired}")

        if len(pnl) > 0:
            print(f"\nPnL médio  : {pnl.mean():+.2f}%")
            print(f"PnL total  : {pnl.sum():+.2f}%")
            print(f"Melhor     : {pnl.max():+.2f}%")
            print(f"Pior       : {pnl.min():+.2f}%")

    open_n = (df["outcome"] == "OPEN").sum()
    if open_n > 0:
        print(f"\nTrades ainda abertos: {open_n}")

    max_candles = pd.to_numeric(df["candles_treinados"], errors="coerce").max()
    print(f"\nMáx candles treino : {max_candles:.0f}")
    print(f"Primeiro sinal     : {df['timestamp'].min()}")
    print(f"Último sinal       : {df['timestamp'].max()}")
    print(f"{'='*55}\n")
