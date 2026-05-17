"""Tests para _stats.py."""
from _stats import freqdist_topk, type_token_ratio


def test_freqdist_topk_basic():
    tokens = ["a", "b", "a", "c", "a", "b"]
    result = freqdist_topk(tokens, k=2)
    assert result == [("a", 3), ("b", 2)]


def test_freqdist_topk_default_k():
    tokens = ["a"] * 100 + ["b"] * 50
    result = freqdist_topk(tokens)
    assert result[0] == ("a", 100)
    assert result[1] == ("b", 50)


def test_type_token_ratio():
    tokens = ["a", "b", "a", "c"]
    assert type_token_ratio(tokens) == 3 / 4


def test_type_token_ratio_empty():
    assert type_token_ratio([]) == 0.0


def test_zipf_fit_recovers_alpha_from_synthetic():
    """Genera corpus con Zipf paramétrico y verifica que zipf_fit lo recupera."""
    import numpy as np
    rng = np.random.default_rng(42)
    raw = rng.zipf(1.5, size=10000)
    tokens = [str(x) for x in raw if x < 1000]
    from _stats import zipf_fit
    alpha, K, r2 = zipf_fit(tokens)
    assert 1.0 < alpha < 2.5
    assert r2 > 0.85


def test_zipf_fit_small_corpus_returns_zeros():
    """Corpus con <10 tipos únicos: devuelve zeros (degenerate)."""
    from _stats import zipf_fit
    alpha, K, r2 = zipf_fit(["a", "b", "c"])
    assert alpha == 0.0 and K == 0.0 and r2 == 0.0


def test_vocab_growth_curve():
    from _stats import vocab_growth_curve
    tokens = ["a", "b", "a", "c", "b", "d"]
    xs, ys = vocab_growth_curve(tokens, stride=1)
    assert ys == [1, 2, 2, 3, 3, 4]
    assert xs == [1, 2, 3, 4, 5, 6]


def test_heaps_fit_on_synthetic():
    """V(N) ~ K·N^β: vocab que crece como sqrt → β ≈ 0.5."""
    tokens = []
    for n in range(1, 10000):
        tokens.append(f"w{int(n ** 0.5)}")
    from _stats import heaps_fit
    beta, K, r2 = heaps_fit(tokens)
    assert 0.3 < beta < 0.7
    assert r2 > 0.85
