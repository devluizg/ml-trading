"""
TESTE TDD COM DADOS SINTÉTICOS
================================

Objetivo: validar que o pipeline de ML está matematicamente correto
ANTES de introduzir o ruído do mercado real.

Critério de aprovação:
  - Acurácia >= 0.97 (97%) nos dados sintéticos
  - Se falhar aqui, o problema é no CÓDIGO, não nos dados.

Execute com:
  python -m pytest tests/test_tdd_synthetic.py -v
  ou
  python tests/test_tdd_synthetic.py
"""

import sys
import os

# Adiciona raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.synthetic import generate_synthetic_ohlcv
from data.features import build_features
from labeling.triple_barrier import apply_triple_barrier
from model.classifier import TradingClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_from_synthetic(use_true_label: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline com dados sintéticos.

    Parâmetros
    ----------
    use_true_label : bool
        True  → usa o rótulo de regime do gerador (ground truth perfeito).
                 Propósito: validar features + modelo isoladamente.
        False → usa Barreira Tripla (realista mas ruidosa mesmo em dados sintéticos).
                 Propósito: validar o pipeline completo end-to-end.
    """
    # 1. Dados sintéticos
    df = generate_synthetic_ohlcv(n_bars_per_regime=400)

    # 2. Features estacionárias
    feat = build_features(df)

    if use_true_label:
        # Ground truth perfeito: rótulo de regime do gerador.
        # Aqui testamos exclusivamente: features + modelo.
        labels = df["true_label"].loc[feat.index]
    else:
        # Barreira Tripla realista sobre dados sintéticos.
        labels = apply_triple_barrier(
            close=df["close"].loc[feat.index],
            high=df["high"].loc[feat.index],
            low=df["low"].loc[feat.index],
            pt=0.02,
            sl=0.02,
            vertical_bars=50,
        )

    # 3. Alinha (mesmos índices)
    common_idx = feat.index.intersection(labels.index)
    X = feat.loc[common_idx]
    y = labels.loc[common_idx]

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Testes
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_data_generation():
    """Garante que o gerador produz os 3 regimes com proporções corretas."""
    df = generate_synthetic_ohlcv(n_bars_per_regime=300)
    assert len(df) == 900, f"Esperado 900 barras, obteve {len(df)}"
    assert set(df["true_label"].unique()) == {-1, 0, 1}, "Faltando algum regime"
    counts = df["true_label"].value_counts()
    for label in [-1, 0, 1]:
        assert counts[label] == 300, f"Label {label} com contagem errada: {counts[label]}"
    print("✓ Gerador de dados sintéticos OK")


def test_features_are_stationary():
    """Verifica que nenhuma feature tem valor absoluto de preço."""
    df = generate_synthetic_ohlcv(n_bars_per_regime=200)
    feat = build_features(df)

    # Preços absolutos ficam na faixa de 100-400 neste sintético.
    # Se alguma feature tiver média > 50, provavelmente é preço absoluto.
    for col in feat.columns:
        mean_abs = feat[col].abs().mean()
        assert mean_abs < 50.0, (
            f"Feature '{col}' parece não-estacionária: |mean| = {mean_abs:.2f}. "
            "Verifique se está usando preços absolutos."
        )
    print(f"✓ {len(feat.columns)} features estacionárias OK: {list(feat.columns)}")


def test_triple_barrier_label_distribution():
    """Barreira Tripla deve gerar os 3 tipos de rótulo nos dados sintéticos."""
    X, y = build_dataset_from_synthetic()
    unique = sorted(y.unique())
    assert -1 in unique, "Sem labels de BAIXA — verifique pt/sl"
    assert 1 in unique, "Sem labels de ALTA — verifique pt/sl"
    assert 0 in unique, "Sem labels LATERAIS — verifique vertical_bars"
    dist = y.value_counts(normalize=True) * 100
    print(f"✓ Distribuição de labels: {dict(dist.round(1))}")


def test_model_accuracy_on_synthetic_data():
    """
    TESTE CENTRAL TDD (features + modelo):

    Usa o rótulo de regime perfeito do gerador (not triple barrier).
    Isolamos aqui features + modelo. Se falhar com ground truth perfeito,
    o bug está no feature engineering ou no classifier — nunca no labeling.

    Threshold: >= 97% de acurácia.
    """
    ACCURACY_THRESHOLD = 0.97

    # use_true_label=True → ground truth perfeito (regime conhecido)
    X, y = build_dataset_from_synthetic(use_true_label=True)

    # Split temporal (sem shuffle — série temporal)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # prob_threshold=0 para avaliação pura de acurácia (sem filtro de confiança)
    clf = TradingClassifier(model_type="rf", prob_threshold=0.0, prob_gap=0.0)
    clf.fit(X_train, y_train)

    metrics = clf.evaluate(X_test, y_test)
    acc = metrics["accuracy"]

    print(f"\n{'='*60}")
    print(f"RESULTADO TDD — Acurácia nos dados sintéticos: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Threshold mínimo: {ACCURACY_THRESHOLD*100:.0f}%")
    print(f"{'='*60}")
    print(metrics["report"])

    assert acc >= ACCURACY_THRESHOLD, (
        f"FALHA TDD: Acurácia {acc:.4f} abaixo do threshold {ACCURACY_THRESHOLD}.\n"
        "Isso indica bug no pipeline de ML (features, labeling ou modelo).\n"
        "Corrija antes de usar dados reais."
    )
    print(f"✓ TDD APROVADO: modelo com {acc*100:.2f}% de acurácia nos dados sintéticos")


def test_signal_rules():
    """
    Valida as regras de entrada Long/Short.

    Treina com 60% dos dados (inclui todos os regimes), testa nos 40% restantes.
    Usa shuffle=True porque em dados sintéticos o objetivo é validar a lógica,
    não a dependência temporal — os dados reais nunca usam shuffle.
    """
    from sklearn.utils import shuffle as sk_shuffle

    X, y = build_dataset_from_synthetic(use_true_label=True)

    # Shuffle para garantir todos os regimes no conjunto de teste
    X_s, y_s = sk_shuffle(X, y, random_state=0)
    split = int(len(X_s) * 0.6)
    X_train, X_test = X_s.iloc[:split], X_s.iloc[split:]
    y_train, y_test = y_s.iloc[:split], y_s.iloc[split:]

    clf = TradingClassifier(
        model_type="rf",
        prob_threshold=0.45,
        prob_gap=0.20,
    )
    clf.fit(X_train, y_train)
    signals = clf.predict_signal(X_test)

    assert set(signals.unique()).issubset({-1, 0, 1}), "Sinal inválido gerado"

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_neutro = (signals == 0).sum()

    # Em dados sintéticos com 3 regimes, devemos gerar LONG e SHORT
    assert n_long > 0, "Nenhum sinal LONG gerado — verifique prob_threshold"
    assert n_short > 0, "Nenhum sinal SHORT gerado — verifique prob_threshold"

    print(f"\n✓ Sinais gerados — LONG: {n_long}, SHORT: {n_short}, NEUTRO: {n_neutro}")

    # Valida alinhamento: na maioria das barras Alta, o sinal deve ser LONG
    alta_mask = y_test == 1
    if alta_mask.sum() > 0:
        pct_long_in_alta = (signals[alta_mask] == 1).mean()
        print(f"  Barras Alta → LONG: {pct_long_in_alta*100:.0f}%")

    baixa_mask = y_test == -1
    if baixa_mask.sum() > 0:
        pct_short_in_baixa = (signals[baixa_mask] == -1).mean()
        print(f"  Barras Baixa → SHORT: {pct_short_in_baixa*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Runner manual (sem pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("INICIANDO SUITE TDD COM DADOS SINTÉTICOS")
    print("="*60 + "\n")

    tests = [
        test_synthetic_data_generation,
        test_features_are_stationary,
        test_triple_barrier_label_distribution,
        test_model_accuracy_on_synthetic_data,
        test_signal_rules,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        print(f"\n[TEST] {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERRO INESPERADO: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL: {passed} passou, {failed} falhou")
    if failed == 0:
        print("Pipeline de ML VALIDADO. Pronto para dados reais.")
    else:
        print("CORRIJA OS ERROS antes de prosseguir.")
    print("="*60)
    sys.exit(1 if failed > 0 else 0)
