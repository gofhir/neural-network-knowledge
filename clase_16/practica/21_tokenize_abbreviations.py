"""21_tokenize_abbreviations.py — Comparativa de tokenizadores en abreviaciones clínicas.

Banco fijo de oraciones por categoría (honoríficos, abreviaciones médicas,
decimales/mediciones, siglas). Cada tokenizador procesa cada oración; la
métrica de interés es n_tokens por oración: valores bajos sugieren que el
tokenizador conserva la abreviación como una unidad, valores altos que la
fragmenta.

Outputs:
  - out/21_abbrev_detail.csv  — fila por (tokenizer, category, sentence).
  - out/21_abbrev_summary.csv — pivot: media de n_tokens por tokenizer × categoría.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _tokenizers import list_tokenizers

OUT_DIR = Path(__file__).parent / "out"

CASES = {
    "honorific": [
        "El Sr. Pérez tiene 45 años.",
        "La Sra. Rodríguez consulta por dolor.",
        "El Dr. González realizó la cirugía.",
    ],
    "abbreviation": [
        "Pte. presenta HTA y DM2.",
        "Dx: NSTEMI.",
        "Tto. con losartán 50 mg/d.",
        "S/o sangrado activo.",
    ],
    "decimal": [
        "Glicemia 145 mg/dL.",
        "PA 140/90 mmHg.",
        "FC 78 lpm.",
    ],
    "siglas": [
        "Paciente con EPOC en tto.",
        "Sospecha de ACV isquémico.",
        "Antecedente de IAM en 2019.",
    ],
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for tok_name, tok in list_tokenizers().items():
        for category, sentences in CASES.items():
            for sent in sentences:
                tokens = tok.tokenize(sent)
                rows.append({
                    "tokenizer": tok_name,
                    "category": category,
                    "sentence": sent,
                    "tokens": str(tokens),
                    "n_tokens": len(tokens),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "21_abbrev_detail.csv", index=False)

    summary = df.groupby(["tokenizer", "category"])["n_tokens"].mean().unstack()
    print("\nMedia de n_tokens por tokenizer × categoría:")
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "21_abbrev_summary.csv")
    print(f"\nOutputs en {OUT_DIR}/")


if __name__ == "__main__":
    main()
