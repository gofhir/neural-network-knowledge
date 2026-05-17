# Práctica NLP Clínico — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construir un pipeline de NLP clásico (NLTK) aplicado a 4 corpora cross-domain (MEDDOCAN, Cantemist, PharmaCoNER, Quijote) para caracterizar texto clínico en español, comparar tokenizadores, descubrir stopwords médicas y demostrar detección PII para FHIR-MDM.

**Architecture:** ~20 scripts numerados secuencialmente + 4 helpers privados (`_corpora.py`, `_stats.py`, `_tokenize.py`, `_eval.py`) + tests con pytest. Helpers proveen interfaces uniformes; scripts experimentales generan figuras y tablas reproducibles en `out/`. Patrón heredado de `clase_14/practica/`.

**Tech Stack:** Python 3.11+ con `uv` para venv. NLTK (tokenizers, stemmers, lemmatizers), HuggingFace `datasets` (carga de corpora), `pandas` (CSVs), `matplotlib` (figuras), `pytest` (tests), `numpy`/`scipy` (estadísticas).

**Design doc:** `clase_16/practica/docs/2026-05-17-practica-nlp-clinico-design.md`

---

## Phase 0: Project skeleton (Tasks 0-2)

### Task 0: Crear estructura mínima + pyproject.toml + .gitignore

**Files:**
- Create: `clase_16/practica/pyproject.toml`
- Create: `clase_16/practica/.gitignore`
- Create: `clase_16/practica/README.md` (esqueleto)
- Create: `clase_16/practica/tests/__init__.py` (vacío)
- Create: `clase_16/practica/tests/conftest.py`

**Step 1: Escribir pyproject.toml**

```toml
[project]
name = "practica-nlp-clinico"
version = "0.1.0"
description = "Práctica clase 16 — NLP clásico sobre corpus clínico"
requires-python = ">=3.11"
dependencies = [
    "nltk>=3.9",
    "datasets>=2.20",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "scipy>=1.12",
    "pyarrow>=15.0",
    "huggingface-hub>=0.24",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

**Step 2: Escribir .gitignore**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/corpora/*.parquet
data/corpora/*.txt
out/
checkpoints/
.hypothesis/
```

**Step 3: Setup venv con uv**

Run:
```bash
cd clase_16/practica
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected: venv creado en `.venv/`, dependencias instaladas sin errores.

**Step 4: Escribir conftest.py**

```python
"""Pytest config compartida."""
import sys
from pathlib import Path

# Permitir imports relativos de _corpora, _stats, etc. desde tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Step 5: Verificar pytest corre (sin tests aún)**

Run: `pytest`
Expected: `no tests ran in X.Xs` (exit 5 — no tests, pero sin errores de imports).

**Step 6: Commit**

```bash
git add clase_16/practica/pyproject.toml \
        clase_16/practica/.gitignore \
        clase_16/practica/README.md \
        clase_16/practica/tests/
git commit -m "chore(clase-16/practica): init project skeleton with pyproject.toml"
```

---

### Task 1: Symlink quijote.txt + carpeta data/corpora

**Files:**
- Create: `clase_16/practica/data/corpora/.gitkeep`
- Create: `clase_16/practica/data/corpora/quijote.txt` (symlink)

**Step 1: Crear carpeta data + symlink**

```bash
cd clase_16/practica
mkdir -p data/corpora out checkpoints
ln -s ../../../clase_14/practica/quijote.txt data/corpora/quijote.txt
touch data/corpora/.gitkeep out/.gitkeep checkpoints/.gitkeep
```

**Step 2: Verificar symlink funciona**

Run: `head -c 200 data/corpora/quijote.txt`
Expected: primeras líneas de Don Quijote ("DON QUIJOTE DE LA MANCHA\nMiguel de Cervantes Saavedra\n...").

**Step 3: Commit**

```bash
git add clase_16/practica/data/ clase_16/practica/out/ clase_16/practica/checkpoints/
git commit -m "chore(clase-16/practica): add data/corpora structure + quijote symlink"
```

---

### Task 2: Helper `_corpora.py` — Doc / Entity dataclasses

**Files:**
- Create: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Step 1: Test failing — dataclasses**

```python
# tests/test_corpora.py
from _corpora import Doc, Entity

def test_doc_dataclass_basic():
    doc = Doc(id="d1", text="hola mundo", source="test",
              annotations=[], metadata={})
    assert doc.id == "d1"
    assert doc.text == "hola mundo"

def test_entity_dataclass_basic():
    e = Entity(start=0, end=5, label="PER", text="Pedro")
    assert e.start == 0 and e.end == 5
    assert e.label == "PER"
```

**Step 2: Verify failing**

Run: `pytest tests/test_corpora.py -v`
Expected: ImportError "No module named '_corpora'".

**Step 3: Implementar dataclasses**

```python
# _corpora.py
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str

@dataclass
class Doc:
    id: str
    text: str
    source: str
    annotations: List[Entity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Step 4: Verify passing**

Run: `pytest tests/test_corpora.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add clase_16/practica/_corpora.py clase_16/practica/tests/test_corpora.py
git commit -m "feat(clase-16/practica): add Doc and Entity dataclasses"
```

---

## Phase 1: Corpus loaders (Tasks 3-7)

### Task 3: `_corpora.py` — `load_quijote()`

**Files:**
- Modify: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Step 1: Test failing**

```python
def test_load_quijote_returns_one_doc():
    from _corpora import load_quijote
    docs = load_quijote()
    assert len(docs) == 1
    assert docs[0].source == "quijote"
    assert len(docs[0].text) > 100_000

def test_load_quijote_text_contains_known_words():
    docs = load_quijote()
    text = docs[0].text.lower()
    assert "quijote" in text
    assert "sancho" in text
    assert "dulcinea" in text
```

**Step 2: Verify failing** — Run: `pytest tests/test_corpora.py::test_load_quijote_returns_one_doc -v`

**Step 3: Implementar**

```python
from pathlib import Path

_CORPORA_DIR = Path(__file__).parent / "data" / "corpora"

def load_quijote() -> List[Doc]:
    path = _CORPORA_DIR / "quijote.txt"
    text = path.read_text(encoding="utf-8")
    return [Doc(id="quijote", text=text, source="quijote",
                annotations=[], metadata={"path": str(path)})]
```

**Step 4: Verify passing** — `pytest tests/test_corpora.py -v` → 4 passed.

**Step 5: Commit**

```bash
git add clase_16/practica/_corpora.py clase_16/practica/tests/test_corpora.py
git commit -m "feat(clase-16/practica): add load_quijote()"
```

---

### Task 4: `_corpora.py` — `load_meddocan()` vía HuggingFace

**Files:**
- Modify: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Context:** MEDDOCAN está en HuggingFace como `bigbio/meddocan`. Requiere `trust_remote_code=True`. El dataset tiene splits `train/validation/test`. Cada example tiene `text`, `entities` con `(offsets, type)` formato BigBio.

**Step 1: Test failing**

```python
def test_load_meddocan_returns_docs():
    from _corpora import load_meddocan
    docs = load_meddocan()
    assert len(docs) >= 800  # combined train+val+test
    assert all(d.source == "meddocan" for d in docs)
    # MEDDOCAN tiene anotaciones PII
    assert any(len(d.annotations) > 0 for d in docs)

def test_load_meddocan_entity_types():
    docs = load_meddocan()
    labels = {ann.label for d in docs for ann in d.annotations}
    # Categorías PII esperadas según MEDDOCAN spec
    assert "NOMBRE_SUJETO_ASISTENCIA" in labels or "NOMBRE" in labels
```

**Step 2: Verify failing** — Run: `pytest tests/test_corpora.py::test_load_meddocan_returns_docs -v`

**Step 3: Implementar**

```python
def load_meddocan() -> List[Doc]:
    """Carga MEDDOCAN desde HuggingFace bigbio/meddocan.

    Combina splits train+validation+test. Convierte annotations BigBio a Entity.
    """
    from datasets import load_dataset

    ds = load_dataset("bigbio/meddocan", "meddocan_bigbio_kb",
                      trust_remote_code=True)

    docs: List[Doc] = []
    for split_name in ["train", "validation", "test"]:
        for example in ds[split_name]:
            entities = []
            for ent in example["entities"]:
                # BigBio format: offsets es lista de [start, end] pairs
                for (start, end), text in zip(ent["offsets"], ent["text"]):
                    entities.append(Entity(
                        start=start, end=end,
                        label=ent["type"], text=text,
                    ))
            # Reconstruir texto desde passages (BigBio convention)
            full_text = "\n".join(p["text"][0] for p in example["passages"])
            docs.append(Doc(
                id=example["document_id"],
                text=full_text,
                source="meddocan",
                annotations=entities,
                metadata={"split": split_name},
            ))
    return docs
```

**Step 4: Verify passing**

Run: `pytest tests/test_corpora.py::test_load_meddocan_returns_docs -v --timeout=300`
Expected: PASS (primera ejecución descarga ~30 MB, ~30s; siguientes son instantáneas por cache HF).

**Step 5: Commit**

```bash
git add clase_16/practica/_corpora.py clase_16/practica/tests/test_corpora.py
git commit -m "feat(clase-16/practica): add load_meddocan() via HuggingFace BigBio"
```

---

### Task 5: `_corpora.py` — `load_cantemist()` + `load_pharmaconer()`

**Files:**
- Modify: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Step 1: Tests failing**

```python
def test_load_cantemist_returns_docs():
    from _corpora import load_cantemist
    docs = load_cantemist()
    assert len(docs) >= 500
    assert all(d.source == "cantemist" for d in docs)
    labels = {ann.label for d in docs for ann in d.annotations}
    assert "MORFOLOGIA_NEOPLASIA" in labels

def test_load_pharmaconer_returns_docs():
    from _corpora import load_pharmaconer
    docs = load_pharmaconer()
    assert len(docs) >= 500
    assert all(d.source == "pharmaconer" for d in docs)
    labels = {ann.label for d in docs for ann in d.annotations}
    assert "NORMALIZABLES" in labels or "PROTEINAS" in labels
```

**Step 2: Verify failing.**

**Step 3: Implementar** (refactorizar load_meddocan a helper común)

```python
def _load_bigbio_kb(dataset_name: str, config: str, source: str) -> List[Doc]:
    """Cargador genérico para datasets BigBio KB format."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, config, trust_remote_code=True)

    docs: List[Doc] = []
    splits = [s for s in ["train", "validation", "test"] if s in ds]
    for split_name in splits:
        for example in ds[split_name]:
            entities = []
            for ent in example["entities"]:
                for (start, end), text in zip(ent["offsets"], ent["text"]):
                    entities.append(Entity(start=start, end=end,
                                           label=ent["type"], text=text))
            full_text = "\n".join(p["text"][0] for p in example["passages"])
            docs.append(Doc(id=example["document_id"], text=full_text,
                            source=source, annotations=entities,
                            metadata={"split": split_name}))
    return docs

def load_meddocan() -> List[Doc]:
    return _load_bigbio_kb("bigbio/meddocan", "meddocan_bigbio_kb", "meddocan")

def load_cantemist() -> List[Doc]:
    return _load_bigbio_kb("bigbio/cantemist", "cantemist_bigbio_kb", "cantemist")

def load_pharmaconer() -> List[Doc]:
    return _load_bigbio_kb("bigbio/pharmaconer", "pharmaconer_bigbio_kb", "pharmaconer")
```

**Step 4: Verify passing** — `pytest tests/test_corpora.py -v` → todos los tests pasan.

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add load_cantemist() and load_pharmaconer()"`.

---

### Task 6: `_corpora.py` — `load_corpus()` dispatcher + `list_corpora()`

**Files:**
- Modify: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Step 1: Tests failing**

```python
def test_list_corpora_returns_known_names():
    from _corpora import list_corpora
    names = list_corpora()
    assert set(names) == {"meddocan", "cantemist", "pharmaconer", "quijote"}

def test_load_corpus_dispatches():
    from _corpora import load_corpus
    docs = load_corpus("quijote")
    assert len(docs) == 1 and docs[0].source == "quijote"

def test_load_corpus_unknown_raises():
    from _corpora import load_corpus
    import pytest
    with pytest.raises(ValueError, match="unknown corpus"):
        load_corpus("nonexistent")
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
_LOADERS = {
    "meddocan": load_meddocan,
    "cantemist": load_cantemist,
    "pharmaconer": load_pharmaconer,
    "quijote": load_quijote,
}

def list_corpora() -> List[str]:
    return list(_LOADERS.keys())

def load_corpus(name: str) -> List[Doc]:
    if name not in _LOADERS:
        raise ValueError(f"unknown corpus: {name!r}. Available: {list(_LOADERS)}")
    return _LOADERS[name]()
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add load_corpus dispatcher + list_corpora"`.

---

### Task 7: `_corpora.py` — Persistencia Parquet (caché)

**Files:**
- Modify: `clase_16/practica/_corpora.py`
- Test: `clase_16/practica/tests/test_corpora.py`

**Step 1: Tests failing**

```python
def test_corpus_persist_and_load(tmp_path, monkeypatch):
    from _corpora import save_corpus, load_corpus_from_cache, Doc, Entity
    docs = [
        Doc(id="d1", text="texto uno", source="test",
            annotations=[Entity(0, 5, "PER", "texto")], metadata={}),
        Doc(id="d2", text="texto dos", source="test", annotations=[], metadata={}),
    ]
    save_corpus(docs, tmp_path / "test.parquet")
    loaded = load_corpus_from_cache(tmp_path / "test.parquet")
    assert len(loaded) == 2
    assert loaded[0].id == "d1"
    assert len(loaded[0].annotations) == 1
    assert loaded[0].annotations[0].label == "PER"
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
import pandas as pd
import json

def save_corpus(docs: List[Doc], path: Path) -> None:
    """Persiste corpus como Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in docs:
        rows.append({
            "id": d.id, "text": d.text, "source": d.source,
            "annotations_json": json.dumps([
                {"start": e.start, "end": e.end, "label": e.label, "text": e.text}
                for e in d.annotations
            ]),
            "metadata_json": json.dumps(d.metadata, default=str),
        })
    pd.DataFrame(rows).to_parquet(path)

def load_corpus_from_cache(path: Path) -> List[Doc]:
    """Lee corpus persistido desde Parquet."""
    df = pd.read_parquet(path)
    docs = []
    for _, row in df.iterrows():
        annotations = [
            Entity(a["start"], a["end"], a["label"], a["text"])
            for a in json.loads(row["annotations_json"])
        ]
        docs.append(Doc(
            id=row["id"], text=row["text"], source=row["source"],
            annotations=annotations,
            metadata=json.loads(row["metadata_json"]),
        ))
    return docs
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add Parquet persistence for corpora"`.

---

## Phase 2: Stats helpers (Tasks 8-11)

### Task 8: `_stats.py` — `freqdist_topk()` + `type_token_ratio()`

**Files:**
- Create: `clase_16/practica/_stats.py`
- Test: `clase_16/practica/tests/test_stats.py`

**Step 1: Tests failing**

```python
# tests/test_stats.py
from _stats import freqdist_topk, type_token_ratio

def test_freqdist_topk_basic():
    tokens = ["a", "b", "a", "c", "a", "b"]
    result = freqdist_topk(tokens, k=2)
    assert result == [("a", 3), ("b", 2)]

def test_type_token_ratio():
    tokens = ["a", "b", "a", "c"]
    assert type_token_ratio(tokens) == 3/4
```

**Step 2: Verify failing** — Run: `pytest tests/test_stats.py -v` → ImportError.

**Step 3: Implementar**

```python
# _stats.py
from collections import Counter
from typing import List, Tuple

def freqdist_topk(tokens: List[str], k: int = 50) -> List[Tuple[str, int]]:
    """Top-k palabras por frecuencia descendente."""
    return Counter(tokens).most_common(k)

def type_token_ratio(tokens: List[str]) -> float:
    """V/N: vocabulario único / tokens totales."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
```

**Step 4: Verify passing** — `pytest tests/test_stats.py -v` → 2 passed.

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add freqdist_topk + type_token_ratio"`.

---

### Task 9: `_stats.py` — `zipf_fit()` con corpus sintético

**Files:**
- Modify: `clase_16/practica/_stats.py`
- Test: `clase_16/practica/tests/test_stats.py`

**Step 1: Test failing**

```python
def test_zipf_fit_recovers_alpha_from_synthetic():
    """Genera corpus con Zipf canónico (α=1) y verifica que zipf_fit lo recupera."""
    import numpy as np
    rng = np.random.default_rng(42)
    # Zipf(1) tiene cola larga: usamos 10k tokens
    raw = rng.zipf(1.5, size=10000)
    tokens = [str(x) for x in raw if x < 1000]  # filtrar outliers
    from _stats import zipf_fit
    alpha, K, r2 = zipf_fit(tokens)
    # zipf con parámetro 1.5 produce alfa cercana a 1.5 en el rank-frequency plot
    assert 1.0 < alpha < 2.5
    assert r2 > 0.85
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
import numpy as np
from scipy.stats import linregress

def zipf_fit(tokens: List[str]) -> Tuple[float, float, float]:
    """Ajusta f(r) = K · r^(-alpha) en log-log usando OLS.

    Returns: (alpha, K, r_squared)
    """
    counts = sorted(Counter(tokens).values(), reverse=True)
    if len(counts) < 10:
        return 0.0, 0.0, 0.0
    ranks = np.arange(1, len(counts) + 1)
    log_r = np.log(ranks)
    log_f = np.log(counts)
    # log(f) = log(K) - alpha * log(r)
    slope, intercept, r_value, _, _ = linregress(log_r, log_f)
    alpha = -slope
    K = np.exp(intercept)
    return float(alpha), float(K), float(r_value ** 2)
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add zipf_fit with OLS log-log regression"`.

---

### Task 10: `_stats.py` — `heaps_fit()` + `vocab_growth_curve()`

**Files:**
- Modify: `clase_16/practica/_stats.py`
- Test: `clase_16/practica/tests/test_stats.py`

**Step 1: Tests failing**

```python
def test_heaps_fit_on_synthetic():
    """V(N) ~ K·N^β en corpus controlado."""
    # Corpus con vocabulario que crece como sqrt: β ≈ 0.5 esperado
    tokens = []
    for n in range(1, 10000):
        # Cada step agrega un token al vocab si n es cuadrado perfecto
        tokens.extend([f"w{int(n**0.5)}"] * 1)
    from _stats import heaps_fit
    beta, K, r2 = heaps_fit(tokens)
    assert 0.3 < beta < 0.7
    assert r2 > 0.85

def test_vocab_growth_curve():
    from _stats import vocab_growth_curve
    tokens = ["a", "b", "a", "c", "b", "d"]
    xs, ys = vocab_growth_curve(tokens, stride=1)
    assert ys == [1, 2, 2, 3, 3, 4]
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
def vocab_growth_curve(tokens: List[str], stride: int = 100) -> Tuple[List[int], List[int]]:
    """Curva V(N): tamaño de vocabulario único a medida que se leen tokens.

    stride: cada cuántos tokens registrar un punto (más eficiente para corpora grandes).
    """
    seen = set()
    xs, ys = [], []
    for i, tok in enumerate(tokens, start=1):
        seen.add(tok)
        if i % stride == 0 or i == len(tokens):
            xs.append(i)
            ys.append(len(seen))
    return xs, ys

def heaps_fit(tokens: List[str]) -> Tuple[float, float, float]:
    """Ajusta V(N) = K · N^beta en log-log."""
    xs, ys = vocab_growth_curve(tokens, stride=max(1, len(tokens) // 200))
    log_n = np.log(xs)
    log_v = np.log(ys)
    slope, intercept, r_value, _, _ = linregress(log_n, log_v)
    beta = slope
    K = np.exp(intercept)
    return float(beta), float(K), float(r_value ** 2)
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add heaps_fit + vocab_growth_curve"`.

---

### Task 11: `_stats.py` — `comparative_plot()` helper

**Files:**
- Modify: `clase_16/practica/_stats.py`
- Test: `clase_16/practica/tests/test_stats.py`

**Step 1: Test failing**

```python
def test_comparative_plot_creates_file(tmp_path):
    from _stats import comparative_plot
    curves = {
        "corpus_a": ([1,2,3,4], [1,2,3,4]),
        "corpus_b": ([1,2,3,4], [1,1,2,2]),
    }
    out = tmp_path / "test.png"
    comparative_plot(curves, title="Test", xlabel="x", ylabel="y",
                     log_x=True, log_y=True, output_path=out)
    assert out.exists() and out.stat().st_size > 1000
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def comparative_plot(curves: Dict[str, Tuple[List, List]],
                     title: str, xlabel: str, ylabel: str,
                     log_x: bool = False, log_y: bool = False,
                     output_path: Path = None) -> Path:
    """Helper para plots comparativos. Devuelve el path donde se guardó."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=name, alpha=0.85, linewidth=1.5)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add comparative_plot helper"`.

---

## Phase 3: Tokenizer wrappers (Tasks 12-13)

### Task 12: `_tokenize.py` — Protocol + 3 wrappers built-in

**Files:**
- Create: `clase_16/practica/_tokenize.py`
- Test: `clase_16/practica/tests/test_tokenize.py`

**Step 1: Tests failing**

```python
# tests/test_tokenize.py
def test_punkt_es_tokenizer_basic():
    from _tokenize import NLTKPunktTokenizer
    tok = NLTKPunktTokenizer(language="spanish")
    sents = tok.sent_tokenize("Hola mundo. ¿Cómo estás?")
    assert len(sents) == 2

def test_treebank_tokenizer_basic():
    from _tokenize import NLTKTreebankTokenizer
    tok = NLTKTreebankTokenizer()
    words = tok.tokenize("Hello world!")
    assert "world" in words and "!" in words

def test_tweet_tokenizer_preserves_emoticons():
    from _tokenize import TweetTokenizer
    tok = TweetTokenizer()
    words = tok.tokenize("Hi :-) #yolo")
    assert ":-)" in words
    assert "#yolo" in words

def test_list_tokenizers_returns_at_least_three():
    from _tokenize import list_tokenizers
    toks = list_tokenizers()
    assert "punkt_es" in toks
    assert "treebank" in toks
    assert "tweet" in toks
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
# _tokenize.py
from typing import Dict, List, Protocol
from pathlib import Path
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer as NLTKTweet
from nltk.tokenize.punkt import PunktSentenceTokenizer

class Tokenizer(Protocol):
    name: str
    def tokenize(self, text: str) -> List[str]: ...
    def sent_tokenize(self, text: str) -> List[str]: ...


class NLTKPunktTokenizer:
    def __init__(self, language: str = "spanish"):
        self.language = language
        self.name = f"punkt_{language[:2]}"

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text, language=self.language)

    def tokenize(self, text: str) -> List[str]:
        # word-level: usa Treebank dentro de cada oración
        return word_tokenize(text, language=self.language)


class NLTKTreebankTokenizer:
    name = "treebank"

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)


class TweetTokenizer:
    name = "tweet"
    def __init__(self):
        self._tk = NLTKTweet()

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return self._tk.tokenize(text)


def list_tokenizers() -> Dict[str, Tokenizer]:
    return {
        "punkt_es": NLTKPunktTokenizer(language="spanish"),
        "punkt_en": NLTKPunktTokenizer(language="english"),
        "treebank": NLTKTreebankTokenizer(),
        "tweet": TweetTokenizer(),
    }
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add tokenizer wrappers"`.

---

### Task 13: `_tokenize.py` — `CustomPunktTokenizer` para modelos entrenados

**Files:**
- Modify: `clase_16/practica/_tokenize.py`
- Test: `clase_16/practica/tests/test_tokenize.py`

**Step 1: Test failing**

```python
def test_custom_punkt_loads_and_tokenizes(tmp_path):
    """Entrena un Punkt mínimo y luego lo carga vía CustomPunktTokenizer."""
    from nltk.tokenize.punkt import PunktTrainer
    trainer = PunktTrainer()
    trainer.train("Esta es una frase. Otra frase aquí. Y otra más.")
    model_path = tmp_path / "custom_punkt.pickle"
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(trainer.get_params(), f)

    from _tokenize import CustomPunktTokenizer
    tok = CustomPunktTokenizer(model_path=model_path, name="test_custom")
    sents = tok.sent_tokenize("Esta es una frase. Otra frase aquí.")
    assert len(sents) == 2
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
class CustomPunktTokenizer:
    def __init__(self, model_path: Path, name: str = "punkt_custom"):
        self.name = name
        self.model_path = Path(model_path)
        with open(self.model_path, "rb") as f:
            params = pickle.load(f)
        self._tk = PunktSentenceTokenizer(train_text=None, lang_vars=None)
        self._tk._params = params

    def sent_tokenize(self, text: str) -> List[str]:
        return self._tk.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        # word-level: fallback al Treebank inglés
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add CustomPunktTokenizer for trained models"`.

---

## Phase 4: Evaluation helpers (Task 14)

### Task 14: `_eval.py` — `precision_recall_f1()` + `sentence_boundary_f1()`

**Files:**
- Create: `clase_16/practica/_eval.py`
- Test: `clase_16/practica/tests/test_eval.py`

**Step 1: Tests failing**

```python
# tests/test_eval.py
from _eval import precision_recall_f1, sentence_boundary_f1
from _corpora import Entity

def test_precision_recall_f1_perfect_match():
    pred = [Entity(0, 5, "PER", "Pedro"), Entity(10, 15, "LOC", "Lima")]
    gold = [Entity(0, 5, "PER", "Pedro"), Entity(10, 15, "LOC", "Lima")]
    result = precision_recall_f1(pred, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

def test_precision_recall_f1_no_overlap():
    pred = [Entity(0, 5, "PER", "Pedro")]
    gold = [Entity(20, 25, "PER", "Otro")]
    result = precision_recall_f1(pred, gold)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0

def test_sentence_boundary_f1_basic():
    pred = ["Hola.", "Mundo."]
    gold = ["Hola.", "Mundo."]
    assert sentence_boundary_f1(pred, gold) == 1.0
```

**Step 2: Verify failing.**

**Step 3: Implementar**

```python
# _eval.py
from typing import Dict, List
from _corpora import Entity

def precision_recall_f1(predicted: List[Entity], gold: List[Entity],
                        match_mode: str = "exact") -> Dict[str, float]:
    """Calcula P, R, F1 sobre listas de entidades.

    match_mode:
      - "exact": misma (start, end, label)
      - "partial": overlap > 0 con mismo label
      - "type_only": solo cuenta el label
    """
    def matches(p: Entity, g: Entity) -> bool:
        if p.label != g.label:
            return False
        if match_mode == "exact":
            return p.start == g.start and p.end == g.end
        elif match_mode == "partial":
            return not (p.end <= g.start or g.end <= p.start)
        elif match_mode == "type_only":
            return True
        else:
            raise ValueError(f"unknown match_mode: {match_mode}")

    tp = sum(1 for p in predicted if any(matches(p, g) for g in gold))
    fp = len(predicted) - tp
    fn = len(gold) - sum(1 for g in gold if any(matches(p, g) for p in predicted))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def sentence_boundary_f1(predicted: List[str], gold: List[str]) -> float:
    """F1 entre listas de oraciones (set-based)."""
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

**Step 4: Verify passing.**

**Step 5: Commit** — `git commit -m "feat(clase-16/practica): add precision_recall_f1 + sentence_boundary_f1"`.

---

## Phase 5: Setup scripts (Tasks 15-17)

### Task 15: `00_setup_env.py` — Verificar deps + descargar NLTK data

**Files:**
- Create: `clase_16/practica/00_setup_env.py`

**Step 1: Implementar**

```python
"""00_setup_env.py — Verificar deps + descargar nltk_data mínimo.

Ejecuta una vez al inicio de cada nueva máquina.
Sale 0 si todo OK, 1 si falta algo.
"""
import sys
import nltk

REQUIRED_NLTK = [
    "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
    "averaged_perceptron_tagger",
]

REQUIRED_PYPI = ["datasets", "pandas", "matplotlib", "numpy", "scipy", "pyarrow"]


def check_pypi():
    failed = []
    for pkg in REQUIRED_PYPI:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            failed.append(pkg)
            print(f"  ✗ {pkg} (MISSING)")
    return failed


def download_nltk():
    for resource in REQUIRED_NLTK:
        print(f"  Downloading {resource}...")
        nltk.download(resource, quiet=True)


if __name__ == "__main__":
    print("=== Verificando dependencias PyPI ===")
    missing = check_pypi()
    if missing:
        print(f"\nERROR: Falta instalar: {missing}")
        print("Run: uv pip install -e \".[dev]\"")
        sys.exit(1)

    print("\n=== Descargando NLTK data ===")
    download_nltk()

    print("\n✓ Setup completo")
    sys.exit(0)
```

**Step 2: Run smoke test**

```bash
cd clase_16/practica && python 00_setup_env.py
```
Expected: lista verde de checks + "Setup completo" + exit code 0.

**Step 3: Commit**

```bash
git add clase_16/practica/00_setup_env.py
git commit -m "feat(clase-16/practica): add 00_setup_env.py"
```

---

### Task 16: `01_download_corpora.py` — Persistir 4 corpora

**Files:**
- Create: `clase_16/practica/01_download_corpora.py`

**Step 1: Implementar**

```python
"""01_download_corpora.py — Descarga y persiste 4 corpora como Parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from _corpora import list_corpora, load_corpus, save_corpus, _CORPORA_DIR


def main():
    for name in list_corpora():
        out_path = _CORPORA_DIR / f"{name}.parquet"
        if out_path.exists():
            print(f"[{name}] Ya existe ({out_path}), skipping. "
                  f"Borrar manualmente para re-descargar.")
            continue
        print(f"[{name}] Cargando...")
        docs = load_corpus(name)
        if name != "quijote":  # Quijote es symlink, no necesita Parquet
            save_corpus(docs, out_path)
            print(f"  ✓ {len(docs)} docs persistidos en {out_path}")
        else:
            print(f"  ✓ Quijote: 1 doc desde symlink")


if __name__ == "__main__":
    main()
```

**Step 2: Run smoke test**

```bash
python 01_download_corpora.py
```
Expected: 4 corpora cargados, primera vez descargan ~30 MB cada uno.

**Step 3: Commit** — `git commit -m "feat(clase-16/practica): add 01_download_corpora.py"`.

---

### Task 17: `02_explore_corpora.py` — Sample + N/V/TTR

**Files:**
- Create: `clase_16/practica/02_explore_corpora.py`

**Step 1: Implementar**

```python
"""02_explore_corpora.py — Estadísticas básicas + sample de cada corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
from _corpora import list_corpora, load_corpus
from _stats import type_token_ratio

OUT_PATH = Path(__file__).parent / "out" / "02_summary.md"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 02 — Resumen de corpora\n",
             "| Corpus | Docs | Tokens | Vocab | TTR | Entidades anotadas |",
             "|---|---|---|---|---|---|"]

    for name in list_corpora():
        docs = load_corpus(name)
        all_text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(all_text)
        n_tokens = len(tokens)
        n_vocab = len(set(tokens))
        ttr = type_token_ratio(tokens)
        n_entities = sum(len(d.annotations) for d in docs)

        print(f"\n=== {name} ===")
        print(f"  Docs: {len(docs)}  N: {n_tokens}  V: {n_vocab}  TTR: {ttr:.4f}  Entidades: {n_entities}")
        print(f"  Sample (primeros 200 chars):\n    {docs[0].text[:200]!r}")

        lines.append(f"| {name} | {len(docs)} | {n_tokens:,} | {n_vocab:,} | "
                     f"{ttr:.4f} | {n_entities:,} |")

    OUT_PATH.write_text("\n".join(lines))
    print(f"\n✓ Resumen escrito en {OUT_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Run smoke test**

```bash
python 02_explore_corpora.py
```
Expected: tabla con 4 filas, archivo `out/02_summary.md` creado.

**Step 3: Commit** — `git commit -m "feat(clase-16/practica): add 02_explore_corpora.py"`.

---

## Phase 6: Descriptivos (Tasks 18-22)

### Task 18: `10_zipf_4corpora.py`

**Files:**
- Create: `clase_16/practica/10_zipf_4corpora.py`

**Step 1: Implementar**

```python
"""10_zipf_4corpora.py — Ley de Zipf sobre los 4 corpora."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from collections import Counter
from _corpora import list_corpora, load_corpus
from _stats import zipf_fit, comparative_plot

OUT_DIR = Path(__file__).parent / "out"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    curves = {}

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text.lower())
        alpha, K, r2 = zipf_fit(tokens)
        print(f"{name:12} alpha={alpha:.3f}  K={K:.1f}  r²={r2:.4f}")
        rows.append({"corpus": name, "alpha": alpha, "K": K, "r2": r2,
                     "n_tokens": len(tokens), "n_vocab": len(set(tokens))})

        # curva log-log para plot
        counts = sorted(Counter(tokens).values(), reverse=True)[:1000]
        ranks = list(range(1, len(counts) + 1))
        curves[name] = (ranks, counts)

    pd.DataFrame(rows).to_csv(OUT_DIR / "10_zipf_fit_params.csv", index=False)
    comparative_plot(curves,
                     title="Ley de Zipf — 4 corpora",
                     xlabel="Rango (rank)", ylabel="Frecuencia",
                     log_x=True, log_y=True,
                     output_path=OUT_DIR / "10_zipf_4corpora.png")
    print(f"\n✓ Tabla: {OUT_DIR / '10_zipf_fit_params.csv'}")
    print(f"✓ Plot:  {OUT_DIR / '10_zipf_4corpora.png'}")


if __name__ == "__main__":
    main()
```

**Step 2: Run smoke test** — `python 10_zipf_4corpora.py` → genera PNG + CSV.

**Step 3: Commit** — `git commit -m "feat(clase-16/practica): add 10_zipf_4corpora.py"`.

---

### Task 19: `11_heaps_4corpora.py`

**Files:**
- Create: `clase_16/practica/11_heaps_4corpora.py`

**Step 1: Implementar** (paralelo a Task 18, usando `heaps_fit` y `vocab_growth_curve`)

```python
"""11_heaps_4corpora.py — Ley de Heaps sobre los 4 corpora."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from _corpora import list_corpora, load_corpus
from _stats import heaps_fit, vocab_growth_curve, comparative_plot

OUT_DIR = Path(__file__).parent / "out"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    curves = {}

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text.lower())
        beta, K, r2 = heaps_fit(tokens)
        print(f"{name:12} beta={beta:.3f}  K={K:.2f}  r²={r2:.4f}")
        rows.append({"corpus": name, "beta": beta, "K": K, "r2": r2})

        xs, ys = vocab_growth_curve(tokens, stride=max(1, len(tokens) // 500))
        curves[name] = (xs, ys)

    pd.DataFrame(rows).to_csv(OUT_DIR / "11_heaps_fit_params.csv", index=False)
    comparative_plot(curves,
                     title="Ley de Heaps — 4 corpora",
                     xlabel="Tokens leídos (N)", ylabel="Vocabulario único (V)",
                     log_x=False, log_y=False,
                     output_path=OUT_DIR / "11_heaps_4corpora.png")
    print(f"\n✓ Outputs en {OUT_DIR}/")


if __name__ == "__main__":
    main()
```

**Step 2: Run + Commit** — análogo a Task 18.

---

### Task 20: `12_freqdist_topk.py` — top 50 lado a lado

**Files:**
- Create: `clase_16/practica/12_freqdist_topk.py`

**Step 1: Implementar**

```python
"""12_freqdist_topk.py — Top 50 palabras por corpus + análisis de únicas clínicas."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from _corpora import list_corpora, load_corpus
from _stats import freqdist_topk

OUT_DIR = Path(__file__).parent / "out"
K = 50


def main():
    tops = {}
    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = [t for t in nltk.word_tokenize(text.lower()) if t.isalpha()]
        tops[name] = freqdist_topk(tokens, K)

    # Tabla lado a lado
    df = pd.DataFrame({
        f"{name}_word": [w for w, _ in tops[name]]
        for name in tops
    })
    df.index = [f"rank_{i+1}" for i in range(K)]
    df.to_csv(OUT_DIR / "12_topk_table.csv")

    # Palabras únicas al dominio clínico (vs Quijote)
    clinical_unique = set()
    for clin in ["meddocan", "cantemist", "pharmaconer"]:
        words_clin = {w for w, _ in tops[clin]}
        words_lit = {w for w, _ in tops["quijote"]}
        clinical_unique |= (words_clin - words_lit)

    lines = ["# Top 50 palabras únicas en clínico (no aparecen en top 50 Quijote)\n"]
    for w in sorted(clinical_unique):
        lines.append(f"- `{w}`")
    (OUT_DIR / "12_topk_clinical_unique.md").write_text("\n".join(lines))
    print(f"✓ Outputs en {OUT_DIR}/")


if __name__ == "__main__":
    main()
```

**Step 2: Run + Commit.**

---

### Task 21: `13_dispersion_clinical.py`

**Files:**
- Create: `clase_16/practica/13_dispersion_clinical.py`

**Step 1: Implementar**

```python
"""13_dispersion_clinical.py — Dispersion plots de keywords médicas por corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _corpora import load_corpus

OUT_DIR = Path(__file__).parent / "out"

KEYWORDS = ["paciente", "dolor", "tratamiento", "diagnóstico", "años", "mg"]


def dispersion_plot(tokens, keywords, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, kw in enumerate(keywords):
        positions = [i for i, t in enumerate(tokens) if t.lower() == kw]
        ax.scatter(positions, [j] * len(positions),
                   marker="|", s=80, color="black", alpha=0.6)
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.set_xlabel("Posición en corpus")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    for name in ["meddocan", "cantemist", "pharmaconer"]:
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = nltk.word_tokenize(text)
        out = OUT_DIR / f"13_dispersion_{name}.png"
        dispersion_plot(tokens, KEYWORDS,
                        title=f"Dispersión de keywords médicas en {name}",
                        output_path=out)
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
```

**Step 2: Run + Commit.**

---

### Task 22: `14_concordance_explorer.py`

**Files:**
- Create: `clase_16/practica/14_concordance_explorer.py`

**Step 1: Implementar**

```python
"""14_concordance_explorer.py — KWIC de términos cross-corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
from nltk.text import Text
from _corpora import list_corpora, load_corpus

OUT_DIR = Path(__file__).parent / "out"
TERMS = ["paciente", "dolor", "tratamiento"]


def main():
    for term in TERMS:
        out_path = OUT_DIR / f"14_concordance_{term}.txt"
        with out_path.open("w") as f:
            for name in list_corpora():
                docs = load_corpus(name)
                text_full = "\n".join(d.text for d in docs)
                tokens = nltk.word_tokenize(text_full)
                tx = Text(tokens)
                f.write(f"\n{'='*60}\n=== {name} — concordance for {term!r}\n{'='*60}\n")
                # Redirigir el output de concordance al archivo
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    tx.concordance(term, width=80, lines=10)
                f.write(buf.getvalue())
        print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run + Commit.**

---

## Phase 7: Tokenización comparada (Tasks 23-26)

### Task 23: `20_punkt_es_vs_en.py`

**Files:** Create `clase_16/practica/20_punkt_es_vs_en.py`

```python
"""20_punkt_es_vs_en.py — Diferencias de sent_tokenize ES vs EN en MEDDOCAN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from _corpora import load_corpus
from _tokenize import NLTKPunktTokenizer

OUT_DIR = Path(__file__).parent / "out"


def main():
    docs = load_corpus("meddocan")[:100]
    tok_es = NLTKPunktTokenizer(language="spanish")
    tok_en = NLTKPunktTokenizer(language="english")

    rows = []
    for d in docs:
        n_es = len(tok_es.sent_tokenize(d.text))
        n_en = len(tok_en.sent_tokenize(d.text))
        rows.append({"doc_id": d.id, "n_sent_es": n_es, "n_sent_en": n_en,
                     "diff": n_en - n_es})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "20_punkt_es_vs_en.csv", index=False)
    print(f"\nDiferencias EN-ES: mean={df['diff'].mean():.2f}, "
          f"max={df['diff'].max()}, abs sum={df['diff'].abs().sum()}")
    print(f"✓ {OUT_DIR / '20_punkt_es_vs_en.csv'}")


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 24: `21_tokenize_abbreviations.py`

**Files:** Create `clase_16/practica/21_tokenize_abbreviations.py`

Banco fijo de 30-50 oraciones con abreviaciones clínicas. Cada tokenizador procesa cada una. Métrica: ¿queda `Sr.` como token o se rompe en `["Sr", "."]`? Tabla 4 tokenizadores × 5 categorías.

```python
"""21_tokenize_abbreviations.py — Comparación cuantitativa de tokenizadores en abreviaciones clínicas."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from _tokenize import list_tokenizers

OUT_DIR = Path(__file__).parent / "out"

CASES = {
    "honorific": [
        "El Sr. Pérez tiene 45 años.",
        "La Sra. Rodríguez consulta por dolor.",
        "El Dr. González realizó la cirugía.",
    ],
    "abbreviation": [
        "Pte. presenta HTA y DM2.",
        "Dx: NSTEMI.",
        "Tto. con losartán 50 mg/d.",
        "S/o sangrado activo.",
    ],
    "decimal": [
        "Glicemia 145 mg/dL.",
        "PA 140/90 mmHg.",
        "FC 78 lpm.",
    ],
    "siglas": [
        "Paciente con EPOC en tto.",
        "Sospecha de ACV isquémico.",
        "Antecedente de IAM en 2019.",
    ],
}


def main():
    rows = []
    for tok_name, tok in list_tokenizers().items():
        for category, sentences in CASES.items():
            for sent in sentences:
                tokens = tok.tokenize(sent)
                rows.append({
                    "tokenizer": tok_name,
                    "category": category,
                    "sentence": sent,
                    "tokens": str(tokens),
                    "n_tokens": len(tokens),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "21_abbrev_accuracy.csv", index=False)
    # Pivot para resumen
    summary = df.groupby(["tokenizer", "category"])["n_tokens"].mean().unstack()
    print("\nMedia de tokens por tokenizer × categoría:")
    print(summary)
    summary.to_csv(OUT_DIR / "21_abbrev_summary.csv")


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 25: `22_punkt_train_custom.py`

**Files:** Create `clase_16/practica/22_punkt_train_custom.py`

```python
"""22_punkt_train_custom.py — Entrenar PunktTrainer sobre cada sub-corpus clínico."""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from nltk.tokenize.punkt import PunktTrainer
from _corpora import load_corpus

CHECKPOINTS = Path(__file__).parent / "checkpoints"
OUT_DIR = Path(__file__).parent / "out"


def main():
    CHECKPOINTS.mkdir(exist_ok=True)
    for name in ["meddocan", "cantemist", "pharmaconer"]:
        docs = load_corpus(name)
        train_text = "\n".join(d.text for d in docs[:int(len(docs) * 0.8)])
        trainer = PunktTrainer()
        trainer.train(train_text, verbose=False)
        # Guardar parámetros
        out_pkl = CHECKPOINTS / f"punkt_{name}.pickle"
        with open(out_pkl, "wb") as f:
            pickle.dump(trainer.get_params(), f)
        # Imprimir abreviaciones aprendidas
        learned = sorted(trainer._params.abbrev_types)[:30]
        out_txt = OUT_DIR / f"22_punkt_{name}_learned.txt"
        out_txt.write_text(
            f"Abreviaciones aprendidas para {name} ({len(trainer._params.abbrev_types)} total):\n"
            + "\n".join(f"  {a}" for a in learned)
        )
        print(f"✓ {out_pkl}  +  {out_txt}")


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 26: `23_eval_punkt_systems.py`

**Files:** Create `clase_16/practica/23_eval_punkt_systems.py`

Crear 30-50 splits de oración manuales gold (en `tests/gold_splits.json`), luego comparar los 5 tokenizadores. Reportar F1 por tokenizador × corpus.

```python
"""23_eval_punkt_systems.py — Evaluación cuantitativa de 5 sentence tokenizers."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from _tokenize import NLTKPunktTokenizer, CustomPunktTokenizer
from _eval import sentence_boundary_f1

OUT_DIR = Path(__file__).parent / "out"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
GOLD = Path(__file__).parent / "tests" / "gold_splits.json"


# GOLD format: {"meddocan": [{"text": "...", "sentences": ["s1.", "s2.", ...]}, ...], ...}
DEFAULT_GOLD = {
    "meddocan": [
        {"text": "El Sr. Pérez presenta HTA. Dx: DM2.",
         "sentences": ["El Sr. Pérez presenta HTA.", "Dx: DM2."]},
        # ... agregar manualmente 10-20 más
    ],
}


def main():
    if not GOLD.exists():
        GOLD.parent.mkdir(parents=True, exist_ok=True)
        GOLD.write_text(json.dumps(DEFAULT_GOLD, indent=2, ensure_ascii=False))
        print(f"⚠ Archivo gold creado en {GOLD}. Edítalo para agregar más ejemplos.")

    gold = json.loads(GOLD.read_text())

    tokenizers = {
        "punkt_es": NLTKPunktTokenizer(language="spanish"),
        "punkt_en": NLTKPunktTokenizer(language="english"),
    }
    for corpus in ["meddocan", "cantemist", "pharmaconer"]:
        ckpt = CHECKPOINTS / f"punkt_{corpus}.pickle"
        if ckpt.exists():
            tokenizers[f"punkt_{corpus}"] = CustomPunktTokenizer(ckpt, name=f"punkt_{corpus}")

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
    print(pivot)
    pivot.to_csv(OUT_DIR / "23_punkt_eval_table.csv")


if __name__ == "__main__":
    main()
```

Run + Commit. (Nota: gold_splits.json se va llenando con anotación manual gradual.)

---

## Phase 8: Stop-words y stemming (Tasks 27-31)

### Task 27: `30_stopwords_baseline.py`

**Files:** Create `clase_16/practica/30_stopwords_baseline.py`

```python
"""30_stopwords_baseline.py — Aplicar NLTK stopwords español a cada corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from nltk.corpus import stopwords
from _corpora import list_corpora, load_corpus

OUT_DIR = Path(__file__).parent / "out"


def main():
    sw = set(stopwords.words("spanish"))
    rows = []
    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)
        tokens = [t.lower() for t in nltk.word_tokenize(text) if t.isalpha()]
        n_total = len(tokens)
        n_filtered = sum(1 for t in tokens if t in sw)
        rows.append({
            "corpus": name,
            "n_total": n_total,
            "n_stopwords": n_filtered,
            "pct_stopwords": n_filtered / n_total * 100,
        })
        print(f"{name:12}  total={n_total:>8,}  sw={n_filtered:>7,}  "
              f"({n_filtered/n_total*100:.1f}%)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "30_stopwords_baseline.csv", index=False)


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 28: `31_stopwords_clinical_discover.py`

**Files:** Create `clase_16/practica/31_stopwords_clinical_discover.py`

Calcular palabras frecuentes en clínico que NO están en NLTK español, rankeadas por TF-IDF inverso (más informativo en clínico, menos en literario = candidato stopword clínico).

```python
"""31_stopwords_clinical_discover.py — Stopwords clínicas NO incluidas en NLTK español."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from _corpora import load_corpus

OUT_DIR = Path(__file__).parent / "out"


def main():
    sw_nltk = set(stopwords.words("spanish"))

    # Frecuencias en clínico combinado vs Quijote
    clinical = []
    for c in ["meddocan", "cantemist", "pharmaconer"]:
        clinical.extend("\n".join(d.text for d in load_corpus(c)) for _ in [None])
    clinical_text = "\n".join(clinical)
    clinical_tokens = [t.lower() for t in nltk.word_tokenize(clinical_text) if t.isalpha()]
    clinical_counter = Counter(clinical_tokens)

    quijote_text = "\n".join(d.text for d in load_corpus("quijote"))
    quijote_tokens = [t.lower() for t in nltk.word_tokenize(quijote_text) if t.isalpha()]
    quijote_counter = Counter(quijote_tokens)

    n_clin = sum(clinical_counter.values())
    n_quij = sum(quijote_counter.values())

    # Candidatos: frecuentes en clínico, NO en NLTK, ratio clínico/quijote > 5
    candidates = []
    for word, freq in clinical_counter.most_common(500):
        if word in sw_nltk:
            continue
        freq_clin_norm = freq / n_clin
        freq_quij_norm = (quijote_counter.get(word, 1)) / n_quij
        ratio = freq_clin_norm / freq_quij_norm
        if ratio > 5:
            candidates.append({
                "word": word, "freq_clinical": freq,
                "freq_quijote": quijote_counter.get(word, 0),
                "ratio": ratio,
            })

    df = pd.DataFrame(candidates).sort_values("ratio", ascending=False).head(50)
    df.to_csv(OUT_DIR / "31_stopwords_clinical_candidates.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 29: `32_stem_clinical.py`

**Files:** Create `clase_16/practica/32_stem_clinical.py`

Aplicar Snowball español a 200 términos clínicos (lista fija). Reportar qué stems quedan reconocibles.

```python
"""32_stem_clinical.py — Snowball español aplicado a vocabulario clínico."""
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


def main():
    ss = SnowballStemmer("spanish")
    rows = []
    for term in TERMS:
        stem = ss.stem(term)
        is_recognizable = stem in term[:max(len(stem)+2, 5)]
        rows.append({"term": term, "stem": stem,
                     "is_substring": stem in term,
                     "length_ratio": len(stem) / len(term)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "32_stem_quality.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 30: `33_lemma_omw_compare.py`

**Files:** Create `clase_16/practica/33_lemma_omw_compare.py`

Comparar Snowball ES vs WordNet ES sobre el mismo vocabulario de 200 términos.

```python
"""33_lemma_omw_compare.py — Lemmatization OMW español vs Snowball ES."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn

OUT_DIR = Path(__file__).parent / "out"


def lemmatize_es(word: str) -> str:
    """Busca lemma de la palabra en WordNet español."""
    synsets = wn.synsets(word, lang="spa")
    if not synsets:
        return word  # no encontrada
    for ss in synsets:
        for lemma in ss.lemmas("spa"):
            if lemma.name() != word and len(lemma.name()) < len(word):
                return lemma.name()
    return word


def main():
    from clase_16.practica.32_stem_clinical import TERMS  # type: ignore
    # Workaround: re-define here to avoid import issues
    pass


# (En la práctica, copiar TERMS de 32_ aquí, o importar via path)

if __name__ == "__main__":
    main()
```

(Nota: este script tiene complejidad por la falta de WordNet español rico. Documentar como hallazgo.)

Run + Commit.

---

### Task 31: `34_normalize_pipeline.py`

**Files:** Create `clase_16/practica/34_normalize_pipeline.py`

Pipeline: load → tokenize → stopword filter → stem. Reportar reducción de vocabulario en cada paso para los 4 corpora.

```python
"""34_normalize_pipeline.py — Pipeline integrada con reporte de reducción."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from _corpora import list_corpora, load_corpus

OUT_PATH = Path(__file__).parent / "out" / "34_pipeline_reduction.md"


def main():
    sw = set(stopwords.words("spanish"))
    ss = SnowballStemmer("spanish")

    lines = ["# 34 — Reducción de vocabulario por etapa\n",
             "| Corpus | V₀ (raw) | V₁ (tras lowercase + alpha) | V₂ (tras stopwords) | V₃ (tras stem) |",
             "|---|---|---|---|---|"]

    for name in list_corpora():
        docs = load_corpus(name)
        text = "\n".join(d.text for d in docs)

        tokens_raw = nltk.word_tokenize(text)
        V0 = len(set(tokens_raw))

        tokens_low = [t.lower() for t in tokens_raw if t.isalpha()]
        V1 = len(set(tokens_low))

        tokens_nostop = [t for t in tokens_low if t not in sw]
        V2 = len(set(tokens_nostop))

        tokens_stemmed = [ss.stem(t) for t in tokens_nostop]
        V3 = len(set(tokens_stemmed))

        lines.append(f"| {name} | {V0:,} | {V1:,} | {V2:,} | {V3:,} |")
        print(f"{name:12} V0={V0:,} → V1={V1:,} → V2={V2:,} → V3={V3:,}")

    OUT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
```

Run + Commit.

---

## Phase 9: MDM-FHIR application (Tasks 32-34)

### Task 32: `40_pii_baseline.py`

**Files:** Create `clase_16/practica/40_pii_baseline.py`

Heurísticas regex + FreqDist para detectar PII en MEDDOCAN.

```python
"""40_pii_baseline.py — Baseline NLP-clásico para detección de PII."""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from _corpora import load_corpus, Entity

OUT_DIR = Path(__file__).parent / "out"

# Regex para PII clásico
PATTERNS = {
    "FECHA": [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b",
    ],
    "TELEFONO": [r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b"],
    "EMAIL": [r"\b[\w.-]+@[\w.-]+\.\w{2,}\b"],
    "ID": [
        r"\b[A-Z]\d{8}[A-Z]?\b",   # DNI/NIE español
        r"\bNHC\s*[:\-]?\s*\d+\b",  # Historia clínica
    ],
}


def detect_pii(text: str) -> list:
    entities = []
    for label, regexes in PATTERNS.items():
        for pat in regexes:
            for m in re.finditer(pat, text):
                entities.append(Entity(
                    start=m.start(), end=m.end(),
                    label=label, text=m.group(),
                ))
    return entities


def main():
    docs = load_corpus("meddocan")
    rows = []
    for d in docs[:200]:
        pred = detect_pii(d.text)
        rows.append({
            "doc_id": d.id,
            "n_predicted": len(pred),
            "n_gold": len(d.annotations),
            "categories_predicted": ",".join(sorted({e.label for e in pred})),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "40_pii_predictions.csv", index=False)
    print(f"✓ {OUT_DIR / '40_pii_predictions.csv'}")


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 33: `41_pii_eval_meddocan.py`

**Files:** Create `clase_16/practica/41_pii_eval_meddocan.py`

Evaluación contra gold MEDDOCAN: Precision/Recall/F1 por categoría.

```python
"""41_pii_eval_meddocan.py — Evaluación de baseline PII contra gold MEDDOCAN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from collections import defaultdict
from _corpora import load_corpus
from _eval import precision_recall_f1

# Importar funciones del baseline
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
baseline = import_module("40_pii_baseline")

OUT_DIR = Path(__file__).parent / "out"

# Mapping de categorías baseline → categorías gold MEDDOCAN
LABEL_MAP = {
    "FECHA": ["FECHAS", "FECHA"],
    "TELEFONO": ["TELEFONO", "NUMERO_TELEFONO"],
    "EMAIL": ["CORREO_ELECTRONICO"],
    "ID": ["ID_SUJETO_ASISTENCIA", "ID_ASEGURAMIENTO", "ID_CONTACTO_ASISTENCIAL"],
}


def normalize_label(label: str) -> str:
    """Mapear gold label a categoría unificada."""
    for unified, golds in LABEL_MAP.items():
        if label in golds:
            return unified
    return label  # mantener otros como están


def main():
    docs = load_corpus("meddocan")
    by_category = defaultdict(lambda: {"pred": [], "gold": []})

    for d in docs[:200]:
        pred = baseline.detect_pii(d.text)
        gold = [
            type(g)(start=g.start, end=g.end,
                    label=normalize_label(g.label), text=g.text)
            for g in d.annotations
        ]
        for category in set(LABEL_MAP.keys()):
            by_category[category]["pred"].extend(p for p in pred if p.label == category)
            by_category[category]["gold"].extend(g for g in gold if g.label == category)

    rows = []
    for category, data in by_category.items():
        metrics = precision_recall_f1(data["pred"], data["gold"], match_mode="partial")
        metrics["category"] = category
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "41_pii_eval.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
```

Run + Commit.

---

### Task 34: `42_mdm_blocker_demo.py`

**Files:** Create `clase_16/practica/42_mdm_blocker_demo.py`

Demo end-to-end: cargar 100 docs MEDDOCAN → normalizar nombres detectados → generar bloques candidatos.

```python
"""42_mdm_blocker_demo.py — Demo: extraer nombres + normalizar + blocking MDM."""
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from collections import defaultdict
from _corpora import load_corpus


def normalize_name(name: str) -> str:
    """Normalización canónica para blocking MDM."""
    # Quitar acentos
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Lowercase + alfanuméricos
    return "".join(c for c in name.lower() if c.isalnum() or c.isspace()).strip()


def soundex_block(name: str) -> str:
    """Blocking key: primeras 3 letras + número de palabras."""
    parts = name.split()
    if not parts:
        return ""
    first = normalize_name(parts[0])
    return f"{first[:3]}_{len(parts)}"


def main():
    docs = load_corpus("meddocan")[:100]
    blocks = defaultdict(list)

    for d in docs:
        # Extraer nombres anotados
        names = [a.text for a in d.annotations if a.label == "NOMBRE_SUJETO_ASISTENCIA"]
        for n in names:
            key = soundex_block(n)
            blocks[key].append({"doc_id": d.id, "name": n,
                                "normalized": normalize_name(n)})

    out = Path(__file__).parent / "out" / "42_mdm_demo.md"
    lines = ["# 42 — MDM Blocker Demo\n",
             "Bloques generados desde 100 docs MEDDOCAN. Cada bloque "
             "contiene nombres candidatos a comparar.\n"]

    for key, items in sorted(blocks.items()):
        if len(items) < 2:
            continue
        lines.append(f"\n## Bloque `{key}` ({len(items)} candidatos)\n")
        for it in items[:5]:
            lines.append(f"- doc={it['doc_id']}: `{it['name']}` → `{it['normalized']}`")

    out.write_text("\n".join(lines))
    print(f"✓ {out}")
    print(f"  {len(blocks)} bloques generados, "
          f"{sum(1 for v in blocks.values() if len(v) >= 2)} con ≥2 candidatos.")


if __name__ == "__main__":
    main()
```

Run + Commit.

---

## Phase 10: Documentación final (Task 35)

### Task 35: `README.md` con hallazgos consolidados

**Files:**
- Modify: `clase_16/practica/README.md`

**Step 1: Inspeccionar todos los outputs en `out/`** y extraer hallazgos cuantitativos:

```bash
cd clase_16/practica
ls out/
```

**Step 2: Redactar README con secciones**:

1. **Resumen ejecutivo** (3 párrafos): qué se hizo, qué dataset, qué herramientas.
2. **Hallazgos clave** (5-7 bullets cuantitativos):
   - "Zipf α en 4 corpora: 1.0X (MEDDOCAN), 1.0X (Cantemist), 1.0X (PharmaCoNER), 1.0X (Quijote). r² ≥ 0.95 en todos."
   - "Heaps β = 0.5X clínico vs 0.4X literario → el vocabulario clínico crece 12% más rápido por nombres propios y abreviaciones."
   - "Punkt español default vs Punkt entrenado en MEDDOCAN: F1 sentence boundary 0.XX → 0.XX (mejora 5 puntos)."
   - "Stopwords clínicas descubiertas: top 10 = ['paciente', 'años', 'mg', 'presenta', ...]. Estas explican Y% del corpus clínico."
   - "Snowball ES sobre fármacos: 80% mantiene stem reconocible. WordNet español tiene cobertura <30% del vocabulario clínico."
   - "PII baseline regex: F1 = 0.XX para fechas, 0.YY para emails, 0.ZZ para IDs. Comparativo: el sistema BERT clínico (BETO) alcanza ~0.95 en MEDDOCAN."
3. **Cómo ejecutar** (paso a paso).
4. **Estructura de scripts** (índice de los 20 scripts).
5. **Limitaciones** identificadas.
6. **Próximos pasos**.

**Step 3: Commit final**

```bash
git add clase_16/practica/README.md
git commit -m "docs(clase-16/practica): final README with quantitative findings"
```

---

## Resumen del plan

- **35 tasks** distribuidos en 10 fases.
- **6 helpers** privados con tests dedicados.
- **20 scripts experimentales** generando ~30 artefactos en `out/`.
- **3 modelos Punkt custom** entrenados y guardados en `checkpoints/`.
- **Total commits estimados**: ~35 (uno por task).
- **Tiempo estimado**: 5-6 sesiones de 2-3 horas.

---

## Execution Handoff

Plan complete and saved to `clase_16/practica/docs/2026-05-17-practica-nlp-clinico-plan.md`.

Dos opciones de ejecución:

**1. Subagent-Driven (esta sesión)** — Dispatch fresh subagent por task, review entre tasks, iteración rápida. Recomendable si quieres ir aprobando incrementos en tiempo real.

**2. Parallel Session (sesión separada)** — Abres una nueva sesión con executing-plans, ejecución por lotes con checkpoints. Recomendable si vas a delegar trabajo y revisar resultados después.

¿Cuál prefieres?
