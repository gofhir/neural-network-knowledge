"""32_stem_clinical.py — Snowball español aplicado a vocabulario clínico.

Banco fijo de 40 términos clínicos (fármacos, diagnósticos, procedimientos
e inflexiones plurales). Aplica el stemmer Snowball español y reporta:
  - stem por término.
  - si el stem es substring del término.
  - ratio de longitud (stem_len / term_len) — proxy de cuánto se "recorta".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from nltk.stem import SnowballStemmer

OUT_DIR = Path(__file__).parent / "out"

TERMS = [
    # Fármacos
    "losartán", "metformina", "atorvastatina", "omeprazol", "paracetamol",
    "ibuprofeno", "amoxicilina", "enalapril", "amlodipino", "warfarina",
    # Diagnósticos
    "hipertensión", "diabetes", "hipotiroidismo", "neumonía", "bronquitis",
    "gastritis", "anemia", "obesidad", "insuficiencia", "hipertrigliceridemia",
    # Procedimientos
    "endoscopía", "colonoscopía", "ecografía", "tomografía", "resonancia",
    "biopsia", "punción", "intubación", "transfusión", "vacunación",
    # Plurales / inflexiones
    "pacientes", "diagnósticos", "tratamientos", "síntomas", "controles",
    "exámenes", "antibióticos", "diabéticos", "hipertensos", "operados",
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    ss = SnowballStemmer("spanish")
    rows = []
    for term in TERMS:
        stem = ss.stem(term)
        rows.append({
            "term": term,
            "stem": stem,
            "is_substring": stem in term,
            "length_ratio": len(stem) / len(term),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "32_stem_quality.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nMedia length_ratio: {df['length_ratio'].mean():.3f}")
    print(f"Stems que son substring: {df['is_substring'].sum()}/{len(df)}")
    print(f"\n{OUT_DIR / '32_stem_quality.csv'}")


if __name__ == "__main__":
    main()
