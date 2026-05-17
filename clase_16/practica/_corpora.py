"""Loaders y modelos para los 4 corpora de la práctica clase 16."""
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
