"""11_heaps_4corpora.py — Ley de Heaps sobre los 4 corpora.

Ajusta V(N) = K · N^beta en log-log y compara las curvas vocab vs tokens.
Outputs:
  - out/11_heaps_fit_params.csv
  - out/11_heaps_4corpora.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd

from _corpora import list_corpora, load_corpus
from _stats import comparative_plot, heaps_fit, vocab_growth_curve

OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    curves = {}

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text.lower())
        beta, K, r2 = heaps_fit(tokens)
        print(f"{name:12} beta={beta:.3f}  K={K:.2f}  r²={r2:.4f}  "
              f"N={len(tokens):,}")
        rows.append({"corpus": name, "beta": beta, "K": K, "r2": r2,
                     "n_tokens": len(tokens)})

        xs, ys = vocab_growth_curve(tokens, stride=max(1, len(tokens) // 500))
        curves[name] = (xs, ys)

    pd.DataFrame(rows).to_csv(OUT_DIR / "11_heaps_fit_params.csv", index=False)
    comparative_plot(curves,
                     title="Ley de Heaps — 4 corpora",
                     xlabel="Tokens leídos (N)",
                     ylabel="Vocabulario único (V)",
                     log_x=False, log_y=False,
                     output_path=OUT_DIR / "11_heaps_4corpora.png")
    print(f"\nOutputs en {OUT_DIR}/")


if __name__ == "__main__":
    main()
