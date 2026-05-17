"""Estadísticas de corpus: frecuencias, Zipf, Heaps, plots comparativos."""
from collections import Counter
from typing import List, Tuple


def freqdist_topk(tokens: List[str], k: int = 50) -> List[Tuple[str, int]]:
    """Top-k palabras por frecuencia descendente."""
    return Counter(tokens).most_common(k)


def type_token_ratio(tokens: List[str]) -> float:
    """V/N: tamaño de vocabulario único sobre tokens totales."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
