"""Loaders y modelos para los 4 corpora de la práctica clase 16."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

_CORPORA_DIR = Path(__file__).parent / "data" / "corpora"


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


def load_quijote() -> List[Doc]:
    """Carga Don Quijote (1 doc) desde el symlink en data/corpora/."""
    path = _CORPORA_DIR / "quijote.txt"
    text = path.read_text(encoding="utf-8")
    return [Doc(id="quijote", text=text, source="quijote",
                annotations=[], metadata={"path": str(path)})]
