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
