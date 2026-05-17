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


def test_load_meddocan_returns_docs():
    from _corpora import load_meddocan
    docs = load_meddocan()
    assert len(docs) >= 800  # combined train+val+test
    assert all(d.source == "meddocan" for d in docs)
    assert any(len(d.annotations) > 0 for d in docs)


def test_load_meddocan_entity_types():
    from _corpora import load_meddocan
    docs = load_meddocan()
    labels = {ann.label for d in docs for ann in d.annotations}
    assert "NOMBRE_SUJETO_ASISTENCIA" in labels or "NOMBRE" in labels


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


def test_list_corpora_returns_known_names():
    from _corpora import list_corpora
    names = list_corpora()
    assert set(names) == {"meddocan", "cantemist", "pharmaconer", "quijote"}


def test_load_corpus_dispatches():
    from _corpora import load_corpus
    docs = load_corpus("quijote")
    assert len(docs) == 1 and docs[0].source == "quijote"


def test_load_corpus_unknown_raises():
    import pytest
    from _corpora import load_corpus
    with pytest.raises(ValueError, match="unknown corpus"):
        load_corpus("nonexistent")


def test_corpus_persist_and_load(tmp_path):
    from _corpora import Doc, Entity, save_corpus, load_corpus_from_cache
    docs = [
        Doc(id="d1", text="texto uno", source="test",
            annotations=[Entity(0, 5, "PER", "texto")], metadata={}),
        Doc(id="d2", text="texto dos", source="test",
            annotations=[], metadata={"split": "train"}),
    ]
    save_corpus(docs, tmp_path / "test.parquet")
    loaded = load_corpus_from_cache(tmp_path / "test.parquet")
    assert len(loaded) == 2
    assert loaded[0].id == "d1"
    assert len(loaded[0].annotations) == 1
    assert loaded[0].annotations[0].label == "PER"
    assert loaded[1].metadata == {"split": "train"}
