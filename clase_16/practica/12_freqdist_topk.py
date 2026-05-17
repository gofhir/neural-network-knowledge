"""12_freqdist_topk.py — Top 50 palabras por corpus + análisis de únicas clínicas.

Filtra a tokens alfabéticos en lowercase. Outputs:
  - out/12_topk_table.csv: tabla lado a lado de top-50 por corpus.
  - out/12_topk_clinical_unique.md: palabras presentes en top-50 clínico
    pero NO en top-50 Quijote (candidatas a stopwords médicas).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd

from _corpora import list_corpora, load_corpus
from _stats import freqdist_topk

OUT_DIR = Path(__file__).parent / "out"
K = 50


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tops = {}
    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = [t for t in nltk.word_tokenize(text.lower()) if t.isalpha()]
        tops[name] = freqdist_topk(tokens, K)
        print(f"{name:12} top-3: {tops[name][:3]}")

    df = pd.DataFrame({
        f"{name}_word": [w for w, _ in tops[name]] for name in tops
    })
    df.index = [f"rank_{i + 1}" for i in range(K)]
    df.to_csv(OUT_DIR / "12_topk_table.csv")

    clinical_unique: set = set()
    quijote_words = {w for w, _ in tops["quijote"]}
    for clin in ["meddocan", "cantemist", "pharmaconer"]:
        words_clin = {w for w, _ in tops[clin]}
        clinical_unique |= (words_clin - quijote_words)

    lines = ["# Top 50 palabras únicas en clínico\n",
             "Palabras que aparecen en el top-50 de ≥1 corpus clínico "
             "pero NO en el top-50 de Quijote. Candidatas a stopwords médicas.\n"]
    for w in sorted(clinical_unique):
        lines.append(f"- `{w}`")
    (OUT_DIR / "12_topk_clinical_unique.md").write_text("\n".join(lines) + "\n")
    print(f"\nOutputs en {OUT_DIR}/")
    print(f"  Candidatas clínicas únicas: {len(clinical_unique)}")


if __name__ == "__main__":
    main()
