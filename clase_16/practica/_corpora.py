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


def _bio_example_to_doc(example, label_names, source: str, default_id: str) -> Doc:
    """Convierte un ejemplo en formato BIO (tokens + ner_tags) a Doc.

    Reconstruye el texto uniendo tokens con espacios y calcula offsets de
    entidades sobre ese texto reconstruido.
    """
    tokens = example["tokens"]
    ner_tags = example["ner_tags"]

    text_parts: List[str] = []
    offsets: List = []
    cursor = 0
    for tok in tokens:
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        offsets.append((cursor, cursor + len(tok)))
        text_parts.append(tok)
        cursor += len(tok)
    text = "".join(text_parts)

    entities: List[Entity] = []
    i = 0
    while i < len(ner_tags):
        tag_name = label_names[ner_tags[i]]
        if tag_name.startswith("B-"):
            label = tag_name[2:]
            start = offsets[i][0]
            end = offsets[i][1]
            j = i + 1
            while j < len(ner_tags) and label_names[ner_tags[j]] == f"I-{label}":
                end = offsets[j][1]
                j += 1
            entities.append(Entity(start=start, end=end, label=label,
                                   text=text[start:end]))
            i = j
        else:
            i += 1

    doc_id = str(example.get("id", default_id)) or default_id
    return Doc(id=doc_id, text=text, source=source, annotations=entities)


def _load_bio_dataset(dataset_name: str, source: str) -> List[Doc]:
    """Loader genérico para datasets HF con schema tokens + ner_tags (BIO)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    label_names = ds[next(iter(ds))].features["ner_tags"].feature.names

    docs: List[Doc] = []
    for split_name in ds:
        for idx, example in enumerate(ds[split_name]):
            doc = _bio_example_to_doc(
                example, label_names, source,
                default_id=f"{source}_{split_name}_{idx}",
            )
            doc.metadata["split"] = split_name
            docs.append(doc)
    return docs


def load_meddocan() -> List[Doc]:
    """Carga MEDDOCAN desde HuggingFace IIC/meddocan (formato BIO).

    Combina splits train+validation+test. Convierte BIO tags a Entity con
    offsets sobre el texto reconstruido desde tokens.
    """
    return _load_bio_dataset("IIC/meddocan", "meddocan")
