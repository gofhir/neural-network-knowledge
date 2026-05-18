"""22_punkt_train_custom.py — Entrenar PunktTrainer en cada sub-corpus clínico.

Para cada corpus clínico (meddocan, cantemist, pharmaconer):
  - Toma el 80% de los docs como train.
  - Entrena PunktTrainer sobre la concatenación de esos textos.
  - Persiste los parámetros aprendidos en checkpoints/punkt_<corpus>.pickle.
  - Reporta las primeras 30 abreviaciones aprendidas en out/22_punkt_<corpus>_learned.txt.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nltk.tokenize.punkt import PunktTrainer

from _corpora import load_corpus

CHECKPOINTS = Path(__file__).parent / "checkpoints"
OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    CHECKPOINTS.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    for name in ["meddocan", "cantemist", "pharmaconer"]:
        docs = load_corpus(name)
        train_docs = docs[:int(len(docs) * 0.8)]
        train_text = "\n".join(d.text for d in train_docs)
        print(f"[{name}] entrenando Punkt sobre {len(train_docs)} docs "
              f"({len(train_text):,} chars)...")

        trainer = PunktTrainer()
        trainer.train(train_text, verbose=False)

        out_pkl = CHECKPOINTS / f"punkt_{name}.pickle"
        with open(out_pkl, "wb") as f:
            pickle.dump(trainer.get_params(), f)

        learned = sorted(trainer._params.abbrev_types)
        out_txt = OUT_DIR / f"22_punkt_{name}_learned.txt"
        out_txt.write_text(
            f"Abreviaciones aprendidas para {name} "
            f"({len(learned)} total):\n"
            + "\n".join(f"  {a}" for a in learned[:30])
            + ("\n  ..." if len(learned) > 30 else "")
            + "\n"
        )
        print(f"  OK  {out_pkl}  +  {out_txt}  ({len(learned)} abbrev.)")


if __name__ == "__main__":
    main()
