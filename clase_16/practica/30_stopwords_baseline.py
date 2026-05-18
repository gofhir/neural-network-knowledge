"""30_stopwords_baseline.py — Aplicar NLTK stopwords español a cada corpus.

Mide qué porcentaje del corpus es filtrable por la lista de stopwords
genérica de NLTK español. Esto pone un baseline contra el cual comparar
las stopwords clínicas descubiertas en el script siguiente.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from nltk.corpus import stopwords

from _corpora import list_corpora, load_corpus

OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sw = set(stopwords.words("spanish"))
    print(f"NLTK stopwords español: {len(sw)} palabras\n")

    rows = []
    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = [t.lower() for t in nltk.word_tokenize(text) if t.isalpha()]
        n_total = len(tokens)
        n_filtered = sum(1 for t in tokens if t in sw)
        rows.append({
            "corpus": name,
            "n_total": n_total,
            "n_stopwords": n_filtered,
            "pct_stopwords": n_filtered / n_total * 100,
        })
        print(f"{name:12}  total={n_total:>10,}  sw={n_filtered:>9,}  "
              f"({n_filtered / n_total * 100:.2f}%)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "30_stopwords_baseline.csv",
                              index=False)
    print(f"\n{OUT_DIR / '30_stopwords_baseline.csv'}")


if __name__ == "__main__":
    main()
