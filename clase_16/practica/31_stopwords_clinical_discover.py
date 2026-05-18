"""31_stopwords_clinical_discover.py — Stopwords clínicas no incluidas en NLTK.

Descubre palabras frecuentes en corpus clínico (combinado meddocan +
cantemist + pharmaconer) que cumplen:
  1. No están en NLTK stopwords español.
  2. Tienen ratio (freq_clínico_norm / freq_quijote_norm) > 5.

Tales palabras son específicas del dominio médico — candidatas a una lista
extendida de stopwords clínicas para downstream tasks.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from nltk.corpus import stopwords

from _corpora import load_corpus

OUT_DIR = Path(__file__).parent / "out"
RATIO_THRESHOLD = 5.0


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sw_nltk = set(stopwords.words("spanish"))

    clinical_text = "\n".join(
        "\n".join(d.text for d in load_corpus(name))
        for name in ["meddocan", "cantemist", "pharmaconer"]
    )
    clinical_tokens = [t.lower() for t in nltk.word_tokenize(clinical_text)
                       if t.isalpha()]
    clinical_counter = Counter(clinical_tokens)

    quijote_text = "\n".join(d.text for d in load_corpus("quijote"))
    quijote_tokens = [t.lower() for t in nltk.word_tokenize(quijote_text)
                      if t.isalpha()]
    quijote_counter = Counter(quijote_tokens)

    n_clin = sum(clinical_counter.values())
    n_quij = sum(quijote_counter.values())

    candidates = []
    for word, freq in clinical_counter.most_common(500):
        if word in sw_nltk:
            continue
        freq_clin_norm = freq / n_clin
        # Laplace smoothing: si la palabra no está en Quijote, asume freq=1.
        freq_quij_norm = (quijote_counter.get(word) or 1) / n_quij
        ratio = freq_clin_norm / freq_quij_norm
        if ratio > RATIO_THRESHOLD:
            candidates.append({
                "word": word,
                "freq_clinical": freq,
                "freq_quijote": quijote_counter.get(word, 0),
                "ratio": ratio,
            })

    df = pd.DataFrame(candidates).sort_values("ratio", ascending=False).head(50)
    df.to_csv(OUT_DIR / "31_stopwords_clinical_candidates.csv", index=False)
    print(f"Top 50 candidatas a stopword clínica (ratio > {RATIO_THRESHOLD}):\n")
    print(df.to_string(index=False))
    print(f"\n{OUT_DIR / '31_stopwords_clinical_candidates.csv'}")


if __name__ == "__main__":
    main()
