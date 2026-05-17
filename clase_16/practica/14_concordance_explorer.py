"""14_concordance_explorer.py — KWIC (keyword-in-context) cross-corpus.

Para cada término en TERMS, escribe un archivo con 10 líneas de concordancia
desde cada corpus, permitiendo comparar cómo se usa el mismo término entre
dominios (clínico vs literario).
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
from nltk.text import Text

from _corpora import list_corpora, load_corpus

OUT_DIR = Path(__file__).parent / "out"
TERMS = ["paciente", "dolor", "tratamiento"]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    # Cache: tokenizar cada corpus una sola vez
    corpus_tokens = {}
    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        corpus_tokens[name] = nltk.word_tokenize(text)

    for term in TERMS:
        out_path = OUT_DIR / f"14_concordance_{term}.txt"
        with out_path.open("w") as f:
            for name in list_corpora():
                tx = Text(corpus_tokens[name])
                f.write(f"\n{'=' * 60}\n")
                f.write(f"=== {name} — concordance for {term!r}\n")
                f.write(f"{'=' * 60}\n")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    tx.concordance(term, width=80, lines=10)
                f.write(buf.getvalue())
        print(f"{out_path.name}")


if __name__ == "__main__":
    main()
