"""00_setup_env.py — Verificar deps + descargar nltk_data mínimo.

Ejecuta una vez al inicio en cada máquina nueva.
Sale 0 si todo OK, 1 si falta algo.
"""
import sys

import nltk

REQUIRED_NLTK = [
    "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
    "averaged_perceptron_tagger",
]

REQUIRED_PYPI = ["datasets", "pandas", "matplotlib", "numpy", "scipy", "pyarrow"]


def check_pypi() -> list:
    failed = []
    for pkg in REQUIRED_PYPI:
        try:
            __import__(pkg)
            print(f"  OK  {pkg}")
        except ImportError:
            failed.append(pkg)
            print(f"  FAIL {pkg} (MISSING)")
    return failed


def download_nltk() -> None:
    for resource in REQUIRED_NLTK:
        print(f"  Downloading {resource}...")
        nltk.download(resource, quiet=True)


if __name__ == "__main__":
    print("=== Verificando dependencias PyPI ===")
    missing = check_pypi()
    if missing:
        print(f"\nERROR: Falta instalar: {missing}")
        print('Run: uv pip install -e ".[dev]"')
        sys.exit(1)

    print("\n=== Descargando NLTK data ===")
    download_nltk()

    print("\nSetup completo.")
    sys.exit(0)
