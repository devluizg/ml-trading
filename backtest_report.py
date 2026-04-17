"""
Relatório de backtest mensal com reset mensal de $50.

Cada mês começa com $50. Se positivo, você saca o lucro.
Se negativo, você deposita a diferença. Sempre volta para $50.

Uso:
  python3 backtest_report.py                       # BTC 15m
  python3 backtest_report.py --config config_eth.yaml
  python3 backtest_report.py --both
"""

import argparse
import logging
import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

from data.storage import load_history
from backtest.walk_forward import run_walk_forward

START_BAL    = 50.0
RISK_PCT     = 0.10
SL_APPROX    = 0.02
TAKER_FEE    = 0.0004
MAX_LOSSES   = 2        # após N perdas seguidas, reduz risco para metade


def simulate_month(trades: pd.Series) -> dict:
    """Simula um mês começando com $50, risco adaptativo."""
    bal           = START_BAL
    consec_losses = 0
    total_fees    = 0.0
    wins = losses = 0

    for ret in trades:
        risk = RISK_PCT * (0.5 if consec_losses >= MAX_LOSSES else 1.0)
        leverage  = risk / SL_APPROX
        notional  = bal * leverage
        fee       = notional * TAKER_FEE * 2
        trade_pnl = ret * notional - fee
        bal          += trade_pnl
        total_fees   += fee
        if ret > 0:
            wins += 1
            consec_losses = 0
        else:
            losses += 1
            consec_losses += 1

    return {
        "end_bal":  round(bal, 2),
        "pnl_usd":  round(bal - START_BAL, 2),
        "fee_usd":  round(total_fees, 2),
        "wins":     wins,
        "losses":   losses,
        "trades":   wins + losses,
    }


def monthly_report(pnl: pd.Series, symbol: str) -> None:
    if pnl.empty:
        print(f"\n{symbol}: nenhum trade gerado.")
        return

    pnl.index = pd.to_datetime(pnl.index, utc=True)
    months = pnl.groupby(pnl.index.to_period("M"))

    print(f"\n{'='*65}")
    print(f"  {symbol}  |  Reset mensal: $50  |  Risco: {RISK_PCT*100:.0f}%/trade")
    print(f"  Cada mês começa com $50 — lucro sacado / prejuízo depositado")
    print(f"{'='*65}")
    print(f"{'Mês':<10} {'T':>4} {'W':>4} {'L':>4} {'WR':>5}  {'P&L $':>8}  {'P&L %':>7}  {'Fluxo':>9}")
    print(f"{'-'*65}")

    total_ganho    = 0.0
    total_depositado = 0.0
    resultados     = []

    for period, group in months:
        r    = simulate_month(group)
        wr   = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
        pct  = r["pnl_usd"] / START_BAL * 100
        emoji = "🟢" if r["pnl_usd"] >= 0 else "🔴"

        if r["pnl_usd"] >= 0:
            fluxo = f"+${r['pnl_usd']:.2f} sacado"
            total_ganho += r["pnl_usd"]
        else:
            fluxo = f"-${abs(r['pnl_usd']):.2f} deposto"
            total_depositado += abs(r["pnl_usd"])

        print(f"{emoji} {str(period):<8} {r['trades']:>4} {r['wins']:>4} {r['losses']:>4} "
              f"{wr:>4.0f}%  {r['pnl_usd']:>+8.2f}  {pct:>+6.1f}%  {fluxo}")
        resultados.append(r["pnl_usd"])

    print(f"{'-'*65}")

    n_pos  = sum(1 for x in resultados if x > 0)
    n_neg  = sum(1 for x in resultados if x < 0)
    n_meses = len(resultados)
    lucro_liquido = total_ganho - total_depositado

    print(f"\n📊 Resumo ({n_meses} meses)")
    print(f"  Meses positivos : {n_pos}  |  Meses negativos: {n_neg}")
    print(f"  Total sacado    : +${total_ganho:.2f}")
    print(f"  Total depositado: -${total_depositado:.2f}")
    print(f"  Lucro líquido   : ${lucro_liquido:+.2f}  (sobre ${START_BAL:.0f} inicial)")
    avg = sum(resultados) / n_meses if n_meses else 0
    print(f"  Média mensal    : {avg:+.2f} USD/mês")
    print(f"  Win Rate global : {pnl[pnl > 0].count() / len(pnl) * 100:.1f}%")
    print(f"\n💡 Anti-martingale: risco cai para 5% após {MAX_LOSSES} perdas seguidas no mês")


def run(config_path: str) -> None:
    config    = yaml.safe_load(open(config_path))
    symbol    = config["symbol"]
    timeframe = config["timeframe"]

    print(f"\nCarregando {symbol} {timeframe}...")
    df = load_history(symbol, timeframe)
    if df.empty:
        print("Sem dados.")
        return

    print(f"Período: {df.index[0].date()} → {df.index[-1].date()} ({len(df):,} candles)")
    print("Rodando backtest walk-forward...")

    result = run_walk_forward(df, config, n_splits=5, verbose=False)
    monthly_report(result.get("pnl", pd.Series()), f"{symbol} {timeframe}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    if args.both:
        run("config.yaml")
        run("config_eth.yaml")
    else:
        run(args.config)


if __name__ == "__main__":
    main()
