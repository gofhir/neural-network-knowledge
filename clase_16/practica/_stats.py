"""Estadísticas de corpus: frecuencias, Zipf, Heaps, plots comparativos."""
from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.stats import linregress


def freqdist_topk(tokens: List[str], k: int = 50) -> List[Tuple[str, int]]:
    """Top-k palabras por frecuencia descendente."""
    return Counter(tokens).most_common(k)


def type_token_ratio(tokens: List[str]) -> float:
    """V/N: tamaño de vocabulario único sobre tokens totales."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def zipf_fit(tokens: List[str]) -> Tuple[float, float, float]:
    """Ajusta f(r) = K · r^(-alpha) en log-log con OLS.

    Returns: (alpha, K, r_squared). Para corpora pequeños (<10 tipos únicos)
    devuelve (0, 0, 0).
    """
    counts = sorted(Counter(tokens).values(), reverse=True)
    if len(counts) < 10:
        return 0.0, 0.0, 0.0
    ranks = np.arange(1, len(counts) + 1)
    log_r = np.log(ranks)
    log_f = np.log(counts)
    # log(f) = log(K) - alpha * log(r)
    slope, intercept, r_value, _, _ = linregress(log_r, log_f)
    alpha = -slope
    K = float(np.exp(intercept))
    return float(alpha), K, float(r_value ** 2)


def vocab_growth_curve(tokens: List[str], stride: int = 100) -> Tuple[List[int], List[int]]:
    """Curva V(N): vocabulario único acumulado al leer tokens secuencialmente.

    stride: cada cuántos tokens se registra un punto (más eficiente para
    corpora grandes). Siempre se registra también el último token.
    """
    seen: set = set()
    xs: List[int] = []
    ys: List[int] = []
    n = len(tokens)
    for i, tok in enumerate(tokens, start=1):
        seen.add(tok)
        if i % stride == 0 or i == n:
            xs.append(i)
            ys.append(len(seen))
    return xs, ys


def heaps_fit(tokens: List[str]) -> Tuple[float, float, float]:
    """Ajusta V(N) = K · N^beta en log-log con OLS.

    Returns: (beta, K, r_squared).
    """
    xs, ys = vocab_growth_curve(tokens, stride=max(1, len(tokens) // 200))
    log_n = np.log(xs)
    log_v = np.log(ys)
    slope, intercept, r_value, _, _ = linregress(log_n, log_v)
    beta = float(slope)
    K = float(np.exp(intercept))
    return beta, K, float(r_value ** 2)
