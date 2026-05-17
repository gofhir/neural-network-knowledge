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


def test_load_quijote_returns_one_doc():
    from _corpora import load_quijote
    docs = load_quijote()
    assert len(docs) == 1
    assert docs[0].source == "quijote"
    assert len(docs[0].text) > 100_000


def test_load_quijote_text_contains_known_words():
    from _corpora import load_quijote
    docs = load_quijote()
    text = docs[0].text.lower()
    assert "quijote" in text
    assert "sancho" in text
    assert "dulcinea" in text
