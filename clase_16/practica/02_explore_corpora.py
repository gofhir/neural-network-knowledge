"""02_explore_corpora.py — Estadísticas básicas + sample de cada corpus.

Tokeniza con `nltk.word_tokenize` (Punkt+Treebank español). Reporta:
  - número de documentos, tokens (N), vocabulario único (V), TTR (V/N).
  - número de entidades anotadas (gold).
  - sample del primer documento (primeros 200 chars).

Outputs:
  - stdout: tabla legible.
  - out/02_summary.md: tabla Markdown.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk

from _corpora import list_corpora, load_corpus
from _stats import type_token_ratio

OUT_PATH = Path(__file__).parent / "out" / "02_summary.md"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 02 — Resumen de corpora\n",
             "| Corpus | Docs | Tokens | Vocab | TTR | Entidades anotadas |",
             "|---|---|---|---|---|---|"]

    for name in list_corpora():
        docs = load_corpus(name)
        all_text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(all_text)
        n_tokens = len(tokens)
        n_vocab = len(set(tokens))
        ttr = type_token_ratio(tokens)
        n_entities = sum(len(d.annotations) for d in docs)

        print(f"\n=== {name} ===")
        print(f"  Docs: {len(docs)}  N: {n_tokens:,}  V: {n_vocab:,}  "
              f"TTR: {ttr:.4f}  Entidades: {n_entities:,}")
        print(f"  Sample (primeros 200 chars):\n    {docs[0].text[:200]!r}")

        lines.append(
            f"| {name} | {len(docs)} | {n_tokens:,} | {n_vocab:,} | "
            f"{ttr:.4f} | {n_entities:,} |"
        )

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nResumen escrito en {OUT_PATH}")


if __name__ == "__main__":
    main()
