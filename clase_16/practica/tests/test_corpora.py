"""Tests para _corpora.py."""
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
