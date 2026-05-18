"""33_lemma_omw_compare.py — WordNet OMW español vs Snowball ES.

Sobre los mismos 40 términos del banco clínico (importados de 32_stem_clinical),
intenta lematizar con WordNet español (OMW). Reporta:
  - coverage: fracción de términos con al menos un synset en español.
  - lemma: primer lemma alternativo (más corto) si existe; sino el término.
  - stem (Snowball) para contraste.

Hipótesis: WordNet español tiene baja cobertura sobre vocabulario clínico
especializado (fármacos comerciales, abreviaciones, neologismos).
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer

# Reutilizar TERMS de 32_stem_clinical sin duplicar la lista.
TERMS = importlib.import_module("32_stem_clinical").TERMS

OUT_DIR = Path(__file__).parent / "out"


def lemmatize_es(word: str) -> tuple:
    """Devuelve (lemma, found): primer lemma español más corto que el word."""
    synsets = wn.synsets(word, lang="spa")
    if not synsets:
        return word, False
    for ss in synsets:
        for lemma in ss.lemmas("spa"):
            if lemma.name() != word and len(lemma.name()) < len(word):
                return lemma.name(), True
    return word, True  # tiene synset pero sin lemma más corto


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ss = SnowballStemmer("spanish")
    rows = []
    for term in TERMS:
        lemma, found = lemmatize_es(term)
        stem = ss.stem(term)
        rows.append({
            "term": term,
            "lemma_omw": lemma,
            "found_in_omw": found,
            "stem_snowball": stem,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "33_lemma_vs_stem.csv", index=False)
    coverage = df["found_in_omw"].sum() / len(df)
    print(df.to_string(index=False))
    print(f"\nCobertura OMW español sobre vocabulario clínico: "
          f"{coverage:.1%} ({df['found_in_omw'].sum()}/{len(df)})")
    print(f"\n{OUT_DIR / '33_lemma_vs_stem.csv'}")


if __name__ == "__main__":
    main()
