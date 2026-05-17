"""01_download_corpora.py — Descarga y persiste los 4 corpora como Parquet.

Quijote queda como symlink (no se persiste a Parquet porque ya es un .txt
plano en data/corpora/). Los demás se materializan como Parquet para
levantar más rápido en scripts posteriores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _corpora import _CORPORA_DIR, list_corpora, load_corpus, save_corpus


def main() -> None:
    for name in list_corpora():
        out_path = _CORPORA_DIR / f"{name}.parquet"
        if name == "quijote":
            print(f"[{name}] symlink local, no requiere Parquet.")
            continue
        if out_path.exists():
            print(f"[{name}] ya existe ({out_path}), skipping. "
                  "Borra el archivo manualmente para re-descargar.")
            continue
        print(f"[{name}] cargando desde HuggingFace...")
        docs = load_corpus(name)
        save_corpus(docs, out_path)
        print(f"  OK  {len(docs)} docs persistidos en {out_path}")


if __name__ == "__main__":
    main()
