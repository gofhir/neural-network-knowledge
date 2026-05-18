"""41_pii_eval_meddocan.py — Evaluación del baseline PII contra gold MEDDOCAN.

Mapea las categorías del baseline (FECHA, TELEFONO, EMAIL, ID) a las
categorías gold de MEDDOCAN (FECHAS, NUMERO_TELEFONO, CORREO_ELECTRONICO,
ID_*). Reporta precision/recall/F1 por categoría usando match_mode="partial"
(overlap > 0 con mismo label) — adecuado para detección regex donde los
offsets pueden no ser idénticos.
"""
import importlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _corpora import Entity, load_corpus
from _eval import precision_recall_f1

baseline = importlib.import_module("40_pii_baseline")

OUT_DIR = Path(__file__).parent / "out"

# Mapping de categorías baseline → categorías gold MEDDOCAN.
LABEL_MAP = {
    "FECHA": ["FECHAS", "FECHA"],
    "TELEFONO": ["NUMERO_TELEFONO", "NUMERO_FAX"],
    "EMAIL": ["CORREO_ELECTRONICO"],
    "ID": [
        "ID_SUJETO_ASISTENCIA",
        "ID_ASEGURAMIENTO",
        "ID_CONTACTO_ASISTENCIAL",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "ID_TITULACION_PERSONAL_SANITARIO",
    ],
}


def normalize_label(label: str) -> str:
    """Mapear gold label a categoría unificada del baseline."""
    for unified, golds in LABEL_MAP.items():
        if label in golds:
            return unified
    return label


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    docs = load_corpus("meddocan")[:200]
    by_category = defaultdict(lambda: {"pred": [], "gold": []})

    for d in docs:
        pred = baseline.detect_pii(d.text)
        gold = [Entity(start=g.start, end=g.end,
                       label=normalize_label(g.label), text=g.text)
                for g in d.annotations]
        for category in LABEL_MAP:
            by_category[category]["pred"].extend(
                p for p in pred if p.label == category
            )
            by_category[category]["gold"].extend(
                g for g in gold if g.label == category
            )

    rows = []
    for category, data in by_category.items():
        metrics = precision_recall_f1(data["pred"], data["gold"],
                                      match_mode="partial")
        metrics["category"] = category
        metrics["n_pred"] = len(data["pred"])
        metrics["n_gold"] = len(data["gold"])
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[["category", "n_pred", "n_gold", "tp", "fp", "fn",
             "precision", "recall", "f1"]]
    df.to_csv(OUT_DIR / "41_pii_eval.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n{OUT_DIR / '41_pii_eval.csv'}")


if __name__ == "__main__":
    main()
