"""Tests para _eval.py."""
import pytest

from _corpora import Entity
from _eval import precision_recall_f1, sentence_boundary_f1


def test_precision_recall_f1_perfect_match():
    pred = [Entity(0, 5, "PER", "Pedro"), Entity(10, 15, "LOC", "Lima")]
    gold = [Entity(0, 5, "PER", "Pedro"), Entity(10, 15, "LOC", "Lima")]
    result = precision_recall_f1(pred, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_precision_recall_f1_no_overlap():
    pred = [Entity(0, 5, "PER", "Pedro")]
    gold = [Entity(20, 25, "PER", "Otro")]
    result = precision_recall_f1(pred, gold)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_precision_recall_f1_partial_match_mode():
    """Overlap parcial con mismo label: cuenta en modo partial, no en exact."""
    pred = [Entity(0, 10, "PER", "Pedro Diaz")]
    gold = [Entity(5, 12, "PER", "Diaz Lo")]
    assert precision_recall_f1(pred, gold, match_mode="exact")["f1"] == 0.0
    assert precision_recall_f1(pred, gold, match_mode="partial")["f1"] == 1.0


def test_precision_recall_f1_label_mismatch():
    """Mismo span pero distinto label → no cuenta."""
    pred = [Entity(0, 5, "LOC", "Pedro")]
    gold = [Entity(0, 5, "PER", "Pedro")]
    assert precision_recall_f1(pred, gold)["f1"] == 0.0


def test_precision_recall_f1_empty():
    """Listas vacías → métricas 0.0 sin division-by-zero."""
    result = precision_recall_f1([], [])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_precision_recall_f1_unknown_match_mode_raises():
    with pytest.raises(ValueError, match="unknown match_mode"):
        precision_recall_f1([], [], match_mode="foo")


def test_sentence_boundary_f1_perfect():
    assert sentence_boundary_f1(["Hola.", "Mundo."], ["Hola.", "Mundo."]) == 1.0


def test_sentence_boundary_f1_partial():
    """Una oración correcta, una incorrecta → F1 = 0.5."""
    pred = ["Hola.", "Adios."]
    gold = ["Hola.", "Mundo."]
    assert sentence_boundary_f1(pred, gold) == 0.5
