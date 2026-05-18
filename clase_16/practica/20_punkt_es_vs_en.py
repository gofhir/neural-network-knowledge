"""20_punkt_es_vs_en.py — Diferencias de sent_tokenize ES vs EN en MEDDOCAN.

Punkt entrenado en español maneja abreviaciones como "Sr.", "Dr.", "Sra."
diferente al modelo entrenado en inglés. Tomamos los primeros 100 docs
de MEDDOCAN y comparamos el conteo de oraciones detectadas por cada uno.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _corpora import load_corpus
from _tokenizers import NLTKPunktTokenizer

OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    docs = load_corpus("meddocan")[:100]
    tok_es = NLTKPunktTokenizer(language="spanish")
    tok_en = NLTKPunktTokenizer(language="english")

    rows = []
    for d in docs:
        n_es = len(tok_es.sent_tokenize(d.text))
        n_en = len(tok_en.sent_tokenize(d.text))
        rows.append({"doc_id": d.id, "n_sent_es": n_es, "n_sent_en": n_en,
                     "diff": n_en - n_es})

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "20_punkt_es_vs_en.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDiferencias EN-ES sobre 100 docs MEDDOCAN:")
    print(f"  mean diff: {df['diff'].mean():.2f}")
    print(f"  max diff:  {df['diff'].max()}")
    print(f"  min diff:  {df['diff'].min()}")
    print(f"  abs sum:   {df['diff'].abs().sum()}")
    print(f"  docs con n_en > n_es: {(df['diff'] > 0).sum()}/{len(df)}")
    print(f"\n{out_path}")


if __name__ == "__main__":
    main()
