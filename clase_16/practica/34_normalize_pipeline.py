"""34_normalize_pipeline.py — Pipeline integrada con reporte de reducción.

Aplica las etapas de normalización en cascada y reporta el tamaño del
vocabulario tras cada una:
  V₀ — tokens crudos (incluye puntuación y mayúsculas).
  V₁ — tras lowercase + filtro alfabético.
  V₂ — tras quitar stopwords NLTK.
  V₃ — tras stem Snowball.

Outputs:
  - out/34_pipeline_reduction.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from _corpora import list_corpora, load_corpus

OUT_PATH = Path(__file__).parent / "out" / "34_pipeline_reduction.md"


def main() -> None:
    sw = set(stopwords.words("spanish"))
    ss = SnowballStemmer("spanish")

    lines = ["# 34 — Reducción de vocabulario por etapa\n",
             "| Corpus | V₀ raw | V₁ lower+alpha | V₂ −stopwords | V₃ +stem | "
             "V₃/V₀ |",
             "|---|---|---|---|---|---|"]
    rows_print = []

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)

        tokens_raw = nltk.word_tokenize(text)
        V0 = len(set(tokens_raw))

        tokens_low = [t.lower() for t in tokens_raw if t.isalpha()]
        V1 = len(set(tokens_low))

        tokens_nostop = [t for t in tokens_low if t not in sw]
        V2 = len(set(tokens_nostop))

        tokens_stemmed = [ss.stem(t) for t in tokens_nostop]
        V3 = len(set(tokens_stemmed))

        ratio = V3 / V0 if V0 else 0.0
        lines.append(
            f"| {name} | {V0:,} | {V1:,} | {V2:,} | {V3:,} | {ratio:.3f} |"
        )
        rows_print.append((name, V0, V1, V2, V3, ratio))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"{'Corpus':12} {'V0':>8} {'V1':>8} {'V2':>8} {'V3':>8} {'V3/V0':>8}")
    for name, V0, V1, V2, V3, ratio in rows_print:
        print(f"{name:12} {V0:>8,} {V1:>8,} {V2:>8,} {V3:>8,} {ratio:>8.3f}")
    print(f"\n{OUT_PATH}")


if __name__ == "__main__":
    main()
