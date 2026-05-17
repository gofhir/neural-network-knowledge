# Práctica clase 16 — NLP clínico

Pipeline NLP clásico (NLTK) aplicado a 4 corpora cross-domain en español:
**MEDDOCAN** (PII clínico), **Cantemist** (oncología), **PharmaCoNER** (fármacos)
y **Quijote** (literatura) como control.

Estado: en desarrollo. Ver [docs/2026-05-17-practica-nlp-clinico-plan.md](docs/2026-05-17-practica-nlp-clinico-plan.md)
para el plan de implementación y [docs/2026-05-17-practica-nlp-clinico-design.md](docs/2026-05-17-practica-nlp-clinico-design.md)
para el diseño.

## Setup

```bash
cd clase_16/practica
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
python 00_setup_env.py
```

## Tests

```bash
pytest
```
