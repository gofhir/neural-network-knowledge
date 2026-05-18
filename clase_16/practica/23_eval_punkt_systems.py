"""23_eval_punkt_systems.py — Evaluación cuantitativa de sentence tokenizers.

Compara N tokenizadores Punkt contra un gold set de splits manuales en
tests/gold_splits.json. Métrica: sentence_boundary_f1 set-based.

Si gold_splits.json no existe, se siembra con un set inicial pequeño que
puede ampliarse manualmente entre corridas.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _eval import sentence_boundary_f1
from _tokenizers import CustomPunktTokenizer, NLTKPunktTokenizer

OUT_DIR = Path(__file__).parent / "out"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
GOLD = Path(__file__).parent / "tests" / "gold_splits.json"


DEFAULT_GOLD = {
    "meddocan": [
        {"text": "El Sr. Pérez presenta HTA. Dx: DM2.",
         "sentences": ["El Sr. Pérez presenta HTA.", "Dx: DM2."]},
        {"text": "Datos del paciente. Nombre: Juan. Apellidos: García.",
         "sentences": ["Datos del paciente.", "Nombre: Juan.",
                       "Apellidos: García."]},
        {"text": "La Sra. Rodríguez consulta por dolor abdominal. Se solicita ecografía.",
         "sentences": ["La Sra. Rodríguez consulta por dolor abdominal.",
                       "Se solicita ecografía."]},
        {"text": "El Dr. González realizó la cirugía. Sin complicaciones.",
         "sentences": ["El Dr. González realizó la cirugía.",
                       "Sin complicaciones."]},
        {"text": "Paciente de 65 años con antecedente de HTA. Refiere disnea.",
         "sentences": ["Paciente de 65 años con antecedente de HTA.",
                       "Refiere disnea."]},
    ],
    "cantemist": [
        {"text": "Mujer de 67 años con hipotiroidismo. Consulta por disnea progresiva.",
         "sentences": ["Mujer de 67 años con hipotiroidismo.",
                       "Consulta por disnea progresiva."]},
        {"text": "Tras 6 ciclos de quimioterapia, el paciente presenta remisión parcial. Continúa seguimiento.",
         "sentences": ["Tras 6 ciclos de quimioterapia, el paciente presenta remisión parcial.",
                       "Continúa seguimiento."]},
        {"text": "Diagnóstico: carcinoma microcítico de pulmón. Estadio IV.",
         "sentences": ["Diagnóstico: carcinoma microcítico de pulmón.",
                       "Estadio IV."]},
        {"text": "Se indica radioterapia adyuvante. Dosis total 60 Gy.",
         "sentences": ["Se indica radioterapia adyuvante.",
                       "Dosis total 60 Gy."]},
    ],
    "pharmaconer": [
        {"text": "Tratamiento con losartán 50 mg/d. Control en 4 semanas.",
         "sentences": ["Tratamiento con losartán 50 mg/d.",
                       "Control en 4 semanas."]},
        {"text": "Se inicia atorvastatina 20 mg al día. Suspender ibuprofeno.",
         "sentences": ["Se inicia atorvastatina 20 mg al día.",
                       "Suspender ibuprofeno."]},
        {"text": "Reacción adversa a amoxicilina. Cambio a azitromicina.",
         "sentences": ["Reacción adversa a amoxicilina.",
                       "Cambio a azitromicina."]},
        {"text": "Dosis máxima de paracetamol 4 g/día. No exceder.",
         "sentences": ["Dosis máxima de paracetamol 4 g/día.",
                       "No exceder."]},
    ],
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if not GOLD.exists():
        GOLD.parent.mkdir(parents=True, exist_ok=True)
        GOLD.write_text(json.dumps(DEFAULT_GOLD, indent=2, ensure_ascii=False))
        print(f"Archivo gold creado en {GOLD}. Edita para agregar más ejemplos.")

    gold = json.loads(GOLD.read_text())

    tokenizers = {
        "punkt_es": NLTKPunktTokenizer(language="spanish"),
        "punkt_en": NLTKPunktTokenizer(language="english"),
    }
    for corpus in ["meddocan", "cantemist", "pharmaconer"]:
        ckpt = CHECKPOINTS / f"punkt_{corpus}.pickle"
        if ckpt.exists():
            tokenizers[f"punkt_{corpus}"] = CustomPunktTokenizer(
                ckpt, name=f"punkt_{corpus}"
            )

    rows = []
    for tok_name, tok in tokenizers.items():
        for corpus, examples in gold.items():
            f1_total = 0.0
            for ex in examples:
                predicted = tok.sent_tokenize(ex["text"])
                f1_total += sentence_boundary_f1(predicted, ex["sentences"])
            avg_f1 = f1_total / len(examples) if examples else 0.0
            rows.append({"tokenizer": tok_name, "corpus": corpus, "f1": avg_f1})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="tokenizer", columns="corpus", values="f1")
    print("\nF1 por tokenizer × corpus:")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    pivot.to_csv(OUT_DIR / "23_punkt_eval_table.csv")
    print(f"\n{OUT_DIR / '23_punkt_eval_table.csv'}")


if __name__ == "__main__":
    main()
