"""
Dimensionamento de posição baseado em risco.

Três métodos implementados
--------------------------
1. Tamanho fixo         → fallback simples (ORDER_AMOUNT no config)
2. Risco fixo por trade → % do equity / distância ao SL
3. Kelly Fracionário    → usa probabilidades do modelo para otimizar crescimento

Referências
-----------
- Lopez de Prado, cap. 10 (position sizing via bet sizing)
- Kelly, J.L. (1956). "A New Interpretation of Information Rate"
"""

import logging

log = logging.getLogger(__name__)


def calc_trade_cost(
    entry_price: float,
    exit_price: float,
    contracts: float,
    taker_fee: float = 0.0004,
) -> float:
    """
    Custo total da operação (entrada + saída) em USDT.
    taker_fee: 0.04% por lado = 0.08% round-trip na Binance Futures.
    """
    notional_entry = contracts * entry_price
    notional_exit  = contracts * exit_price
    return (notional_entry + notional_exit) * taker_fee


def size_by_fixed_risk(
    equity: float,
    entry_price: float,
    sl_price: float,
    risk_pct: float = 0.01,
    contract_value: float = 1.0,
) -> float:
    """
    Tamanho de posição baseado em risco fixo por trade.

    contratos = (equity * risk_pct) / (|entry - sl| * contract_value)

    Garante que se o SL for atingido, a perda máxima seja `risk_pct` do equity.

    Parâmetros
    ----------
    equity : float
        Capital total disponível (USDT).
    entry_price : float
        Preço de entrada estimado.
    sl_price : float
        Preço do stop loss.
    risk_pct : float
        Fração do equity a arriscar (padrão: 1%).
    contract_value : float
        Valor de 1 contrato em USDT (para contratos lineares = 1.0).

    Retorno
    -------
    float — número de contratos (arredondado para 3 casas decimais).
    """
    risk_amount = equity * risk_pct
    price_distance = abs(entry_price - sl_price)

    if price_distance == 0:
        log.warning("Distância ao SL é zero — usando tamanho mínimo")
        return 0.001

    contracts = risk_amount / (price_distance * contract_value)
    contracts = round(contracts, 3)

    # Limita mínimo e máximo para evitar alavancagem excessiva
    contracts = max(0.001, min(contracts, equity / entry_price * 0.5))
    return contracts


def size_by_kelly(
    equity: float,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    prob_win: float,
    kelly_fraction: float = 0.25,
    contract_value: float = 1.0,
) -> float:
    """
    Tamanho de posição pelo Critério de Kelly Fracionário.

    Kelly clássico: f* = (b*p - q) / b
    onde:
        b = payoff ratio = |tp - entry| / |entry - sl|
        p = probabilidade de ganhar (do modelo)
        q = 1 - p

    Kelly fracionário: multiplica f* por kelly_fraction (conservador).

    Parâmetros
    ----------
    equity : float
        Capital disponível.
    entry_price, sl_price, tp_price : float
        Preços de referência.
    prob_win : float
        Probabilidade de ganhar prevista pelo modelo (0-1).
    kelly_fraction : float
        Fração do Kelly a usar (0.25 = 1/4 Kelly, muito mais conservador).
    contract_value : float
        Valor de 1 contrato.

    Retorno
    -------
    float — número de contratos.
    """
    gain = abs(tp_price - entry_price)
    loss = abs(entry_price - sl_price)

    if loss == 0 or gain == 0:
        return size_by_fixed_risk(equity, entry_price, sl_price)

    b = gain / loss          # payoff ratio
    p = prob_win
    q = 1 - p

    kelly_f = (b * p - q) / b

    if kelly_f <= 0:
        log.info(f"Kelly negativo ({kelly_f:.3f}) — sem entrada recomendada")
        return 0.0

    # Fração do equity a apostar
    bet_fraction = kelly_f * kelly_fraction
    bet_amount = equity * bet_fraction
    contracts = bet_amount / (entry_price * contract_value)
    contracts = round(max(0.001, contracts), 3)

    log.info(
        f"Kelly: b={b:.2f}, p={p:.2f}, f*={kelly_f:.3f}, "
        f"frac={kelly_fraction}, contratos={contracts}"
    )
    return contracts


def get_position_size(
    config: dict,
    equity: float,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    prob_win: float,
) -> float:
    """
    Seleciona o método de sizing conforme config e retorna o tamanho.

    Parâmetros
    ----------
    config : dict
        Seção 'risk' do config.yaml.
    equity : float
        Saldo disponível em USDT.
    entry_price, sl_price, tp_price : float
        Níveis de preço.
    prob_win : float
        Probabilidade de vitória prevista pelo modelo.

    Retorno
    -------
    float — contratos a negociar.
    """
    if equity <= 0:
        log.warning("Equity zero ou negativo — usando tamanho mínimo")
        return config.get("order_amount", 0.001)

    if config.get("use_kelly", False):
        size = size_by_kelly(
            equity=equity,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            prob_win=prob_win,
            kelly_fraction=config.get("kelly_fraction", 0.25),
        )
        if size == 0:
            return 0.0
        return size

    return size_by_fixed_risk(
        equity=equity,
        entry_price=entry_price,
        sl_price=sl_price,
        risk_pct=config.get("risk_per_trade_pct", 0.01),
    )
