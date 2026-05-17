"""10_zipf_4corpora.py — Ley de Zipf sobre los 4 corpora.

Tokeniza con NLTK (lowercased), ajusta f(r) = K · r^(-alpha) en log-log
y guarda:
  - out/10_zipf_fit_params.csv: parámetros (alpha, K, r²) y tamaños.
  - out/10_zipf_4corpora.png: curvas log-log comparadas.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd

from _corpora import list_corpora, load_corpus
from _stats import comparative_plot, zipf_fit

OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    curves = {}

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text.lower())
        alpha, K, r2 = zipf_fit(tokens)
        print(f"{name:12} alpha={alpha:.3f}  K={K:.1f}  r²={r2:.4f}  "
              f"N={len(tokens):,}  V={len(set(tokens)):,}")
        rows.append({"corpus": name, "alpha": alpha, "K": K, "r2": r2,
                     "n_tokens": len(tokens), "n_vocab": len(set(tokens))})

        # Curva log-log: top 1000 ranks (resto invisible en plot)
        counts = sorted(Counter(tokens).values(), reverse=True)[:1000]
        ranks = list(range(1, len(counts) + 1))
        curves[name] = (ranks, counts)

    pd.DataFrame(rows).to_csv(OUT_DIR / "10_zipf_fit_params.csv", index=False)
    comparative_plot(curves,
                     title="Ley de Zipf — 4 corpora",
                     xlabel="Rango (rank)", ylabel="Frecuencia",
                     log_x=True, log_y=True,
                     output_path=OUT_DIR / "10_zipf_4corpora.png")
    print(f"\nTabla: {OUT_DIR / '10_zipf_fit_params.csv'}")
    print(f"Plot:  {OUT_DIR / '10_zipf_4corpora.png'}")


if __name__ == "__main__":
    main()
