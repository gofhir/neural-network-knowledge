"""40_pii_baseline.py — Baseline NLP-clásico para detección de PII.

Aplica reglas regex sobre los primeros 200 docs de MEDDOCAN para detectar
4 categorías de PII: FECHA, TELEFONO, EMAIL, ID. Reporta conteo por
categoría y predicciones por documento.

La función detect_pii() se reutiliza en 41_pii_eval_meddocan.py para
evaluación cuantitativa contra el gold.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _corpora import Entity, load_corpus

OUT_DIR = Path(__file__).parent / "out"

# Regex por categoría PII. Cada categoría puede tener varios patrones.
PATTERNS = {
    "FECHA": [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b",
    ],
    "TELEFONO": [r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b"],
    "EMAIL": [r"\b[\w.-]+@[\w.-]+\.\w{2,}\b"],
    "ID": [
        r"\b[A-Z]\d{8}[A-Z]?\b",     # DNI/NIE español
        r"\bNHC\s*[:\-]?\s*\d+\b",   # Historia clínica
    ],
}


def detect_pii(text: str) -> list:
    """Devuelve lista de Entity con la PII detectada por regex."""
    entities = []
    for label, regexes in PATTERNS.items():
        for pat in regexes:
            for m in re.finditer(pat, text):
                entities.append(Entity(
                    start=m.start(), end=m.end(),
                    label=label, text=m.group(),
                ))
    return entities


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    docs = load_corpus("meddocan")[:200]
    rows = []
    cat_counts = {label: 0 for label in PATTERNS}
    for d in docs:
        pred = detect_pii(d.text)
        for p in pred:
            cat_counts[p.label] += 1
        rows.append({
            "doc_id": d.id,
            "n_predicted": len(pred),
            "n_gold": len(d.annotations),
            "categories_predicted": ",".join(
                sorted({e.label for e in pred})
            ),
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "40_pii_predictions.csv", index=False)
    print(f"Total predicciones sobre 200 docs MEDDOCAN:")
    for cat, n in cat_counts.items():
        print(f"  {cat:10} {n}")
    print(f"\n{OUT_DIR / '40_pii_predictions.csv'}")


if __name__ == "__main__":
    main()
