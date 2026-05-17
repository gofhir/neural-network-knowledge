"""13_dispersion_clinical.py — Dispersion plots de keywords médicas por corpus.

Para cada corpus clínico, marca cada posición donde aparece una keyword
elegida (paciente, dolor, tratamiento, diagnóstico, años, mg). Esto muestra
si los términos se concentran en regiones específicas del corpus o se
distribuyen uniformemente.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk

from _corpora import load_corpus

OUT_DIR = Path(__file__).parent / "out"
KEYWORDS = ["paciente", "dolor", "tratamiento", "diagnóstico", "años", "mg"]


def dispersion_plot(tokens, keywords, title, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, kw in enumerate(keywords):
        positions = [i for i, t in enumerate(tokens) if t.lower() == kw]
        ax.scatter(positions, [j] * len(positions),
                   marker="|", s=80, color="black", alpha=0.6)
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.set_xlabel("Posición en corpus (tokens)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name in ["meddocan", "cantemist", "pharmaconer"]:
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text)
        out = OUT_DIR / f"13_dispersion_{name}.png"
        dispersion_plot(tokens, KEYWORDS,
                        title=f"Dispersión de keywords médicas en {name}",
                        output_path=out)
        counts = {kw: sum(1 for t in tokens if t.lower() == kw)
                  for kw in KEYWORDS}
        print(f"{name:12} {out.name}  counts={counts}")


if __name__ == "__main__":
    main()
