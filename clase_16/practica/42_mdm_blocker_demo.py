"""42_mdm_blocker_demo.py — Demo end-to-end: extraer nombres + normalizar + blocking MDM.

Sobre los primeros 100 docs de MEDDOCAN:
  1. Extrae nombres anotados como NOMBRE_SUJETO_ASISTENCIA (gold).
  2. Aplica normalización canónica (NFD, sin acentos, lowercase, solo alfanuméricos).
  3. Genera un blocking key heurístico (primeras 3 letras + número de palabras).
  4. Agrupa por blocking key. Bloques con ≥2 candidatos son pares candidatos
     a deduplicación en un pipeline MDM real (paso previo a un scorer
     determinístico/GBM).
"""
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _corpora import load_corpus


def normalize_name(name: str) -> str:
    """Normalización canónica: NFD → strip diacríticos → lowercase → solo alfanum/espacio."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return "".join(
        c for c in name.lower() if c.isalnum() or c.isspace()
    ).strip()


def soundex_block(name: str) -> str:
    """Blocking key: primeras 3 letras del primer token + número de palabras.

    No es Soundex real (mismo nombre por simplicidad); el principio es agrupar
    nombres "parecidos en superficie" para reducir comparaciones cuadráticas.
    """
    parts = normalize_name(name).split()
    if not parts:
        return ""
    first = parts[0]
    return f"{first[:3]}_{len(parts)}"


def main() -> None:
    docs = load_corpus("meddocan")[:100]
    blocks: dict = defaultdict(list)

    for d in docs:
        names = [a.text for a in d.annotations
                 if a.label == "NOMBRE_SUJETO_ASISTENCIA"]
        for n in names:
            key = soundex_block(n)
            blocks[key].append({"doc_id": d.id, "name": n,
                                "normalized": normalize_name(n)})

    out = Path(__file__).parent / "out" / "42_mdm_demo.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    candidates_blocks = {k: v for k, v in blocks.items() if len(v) >= 2}
    lines = ["# 42 — MDM Blocker Demo\n",
             f"Sobre 100 docs MEDDOCAN extraídos por la anotación gold "
             f"`NOMBRE_SUJETO_ASISTENCIA`.\n",
             f"- Nombres extraídos: {sum(len(v) for v in blocks.values())}",
             f"- Bloques generados: {len(blocks)}",
             f"- Bloques con ≥2 candidatos (pares de comparación): "
             f"{len(candidates_blocks)}\n"]

    for key, items in sorted(candidates_blocks.items(),
                             key=lambda kv: -len(kv[1]))[:20]:
        lines.append(f"\n## Bloque `{key}` ({len(items)} candidatos)\n")
        for it in items[:5]:
            lines.append(
                f"- doc={it['doc_id']}: `{it['name']}` → `{it['normalized']}`"
            )

    out.write_text("\n".join(lines) + "\n")
    print(f"Nombres extraídos: {sum(len(v) for v in blocks.values())}")
    print(f"Bloques generados: {len(blocks)}")
    print(f"Bloques con ≥2 candidatos: {len(candidates_blocks)}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
