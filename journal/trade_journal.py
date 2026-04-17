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
BALANCE_PATH   = Path(__file__).parent / "balance.json"

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
    "pnl_usd",       # P&L em dólares (já descontando taxas)
    "fee_usd",       # taxas pagas
    "resolved_at",
]

log = logging.getLogger(__name__)


def get_reference_balance() -> float:
    """Retorna saldo de referência (padrão $50 se não configurado)."""
    if BALANCE_PATH.exists():
        data = json.loads(BALANCE_PATH.read_text())
        return float(data.get("balance", 50.0))
    return 50.0


def set_reference_balance(amount: float) -> None:
    """Atualiza saldo de referência (chamado via /saldo no Telegram)."""
    data = {
        "balance": amount,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    BALANCE_PATH.write_text(json.dumps(data, indent=2))
    log.info(f"Saldo de referência atualizado: ${amount:.2f}")


def monthly_summary(month: Optional[str] = None) -> str:
    """
    Retorna resumo mensal formatado para o Telegram.
    month: 'YYYY-MM' ou None para o mês atual.
    """
    df = load_journal()
    if df.empty:
        return "📭 Nenhum trade registrado ainda."

    df = df[df["signal"] != "NEUTRO"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    if month is None:
        month = df["month"].max()

    month_df = df[df["month"] == month]
    resolved = month_df[month_df["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]

    if resolved.empty:
        return f"📅 *{month}*\nNenhum trade resolvido neste mês."

    wins     = (resolved["outcome"] == "WIN").sum()
    losses   = (resolved["outcome"] == "LOSS").sum()
    expired  = (resolved["outcome"] == "EXPIRED").sum()
    total    = len(resolved)
    wr       = wins / total * 100 if total > 0 else 0

    pnl_usd = pd.to_numeric(resolved.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").sum()
    fee_usd = pd.to_numeric(resolved.get("fee_usd", pd.Series(dtype=float)), errors="coerce").sum()

    ref_bal  = get_reference_balance()
    pnl_pct_real = pnl_usd / ref_bal * 100 if ref_bal > 0 else 0
    saldo_final  = ref_bal + pnl_usd

    emoji = "🟢" if pnl_usd >= 0 else "🔴"

    lines = [
        f"📅 <b>{month}</b>  {emoji}",
        f"Trades   : {total}  ({wins}W / {losses}L / {expired}exp)",
        f"Win Rate : {wr:.1f}%",
        f"P&amp;L      : <b>{pnl_usd:+.2f} USD</b>  ({pnl_pct_real:+.1f}%)",
        f"Taxas    : -{fee_usd:.2f} USD",
        f"Saldo ref: ${ref_bal:.2f} → ${saldo_final:.2f}",
    ]
    return "\n".join(lines)


def all_months_summary() -> str:
    """Retorna P&L de todos os meses registrados."""
    df = load_journal()
    if df.empty:
        return "📭 Nenhum trade registrado ainda."

    df = df[df["signal"] != "NEUTRO"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    resolved = df[df["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]

    if resolved.empty:
        return "📭 Nenhum trade resolvido ainda."

    lines = ["📊 <b>Resultado por mês</b>"]
    ref_bal = get_reference_balance()

    for month, grp in resolved.groupby("month"):
        wins   = (grp["outcome"] == "WIN").sum()
        losses = (grp["outcome"] == "LOSS").sum()
        total  = len(grp)
        wr     = wins / total * 100 if total > 0 else 0
        pnl    = pd.to_numeric(grp.get("pnl_usd", pd.Series(dtype=float)), errors="coerce").sum()
        pct    = pnl / ref_bal * 100 if ref_bal > 0 else 0
        emoji  = "🟢" if pnl >= 0 else "🔴"
        lines.append(f"{emoji} {month}: {pnl:+.2f} USD ({pct:+.1f}%)  {wins}W/{losses}L  WR {wr:.0f}%")

    return "\n".join(lines)


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
        "pnl_usd": "",
        "fee_usd": "",
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
    """Carrega o journal como DataFrame, tolerante a colunas faltando."""
    if not JOURNAL_PATH.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(JOURNAL_PATH, parse_dates=["timestamp"], on_bad_lines="skip")
    # Garante todas as colunas esperadas
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Colunas de texto: pandas lê células vazias como NaN (float64);
    # forçar object evita "Invalid value for dtype float64" ao gravar strings
    for col in ["outcome", "exit_price", "resolved_at", "signal", "symbol", "timeframe"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df[_COLUMNS]


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

        # P&L em $ usando saldo de referência e risco de 10%
        ref_bal    = get_reference_balance()
        risk_amt   = ref_bal * 0.10           # 10% do saldo = risco por trade
        sl_dist    = abs(entry_price - float(trade["sl_price"])) / entry_price
        contracts  = (risk_amt / sl_dist / entry_price) if sl_dist > 0 else 0.001
        taker_fee  = 0.0004
        fee_usd    = round((contracts * entry_price + contracts * exit_price) * taker_fee, 4)
        pnl_usd    = round(pnl_pct / 100 * contracts * entry_price - fee_usd, 4)

        updated_rows.at[idx, "outcome"]     = outcome
        updated_rows.at[idx, "exit_price"]  = exit_price
        updated_rows.at[idx, "pnl_pct"]     = pnl_pct
        updated_rows.at[idx, "pnl_usd"]     = pnl_usd
        updated_rows.at[idx, "fee_usd"]     = fee_usd
        updated_rows.at[idx, "resolved_at"] = datetime.now(timezone.utc).isoformat()
        resolved += 1
        log.info(f"Trade resolvido: {signal} @ {entry_price:.2f} → {outcome} ({pnl_pct:+.2f}% / {pnl_usd:+.2f} USD)")

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
