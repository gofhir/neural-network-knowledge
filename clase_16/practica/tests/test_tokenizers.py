"""Tests para _tokenizers.py."""


def test_punkt_es_tokenizer_basic():
    from _tokenizers import NLTKPunktTokenizer
    tok = NLTKPunktTokenizer(language="spanish")
    sents = tok.sent_tokenize("Hola mundo. ¿Cómo estás?")
    assert len(sents) == 2


def test_treebank_tokenizer_basic():
    from _tokenizers import NLTKTreebankTokenizer
    tok = NLTKTreebankTokenizer()
    words = tok.tokenize("Hello world!")
    assert "world" in words and "!" in words


def test_tweet_tokenizer_preserves_emoticons():
    from _tokenizers import TweetTokenizer
    tok = TweetTokenizer()
    words = tok.tokenize("Hi :-) #yolo")
    assert ":-)" in words
    assert "#yolo" in words


def test_list_tokenizers_returns_at_least_three():
    from _tokenizers import list_tokenizers
    toks = list_tokenizers()
    assert "punkt_es" in toks
    assert "treebank" in toks
    assert "tweet" in toks


def test_custom_punkt_loads_and_tokenizes(tmp_path):
    """Entrena un Punkt mínimo, guarda parámetros, los carga vía CustomPunktTokenizer."""
    import pickle
    from nltk.tokenize.punkt import PunktTrainer

    trainer = PunktTrainer()
    trainer.train("Esta es una frase. Otra frase aquí. Y otra más.")
    model_path = tmp_path / "custom_punkt.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(trainer.get_params(), f)

    from _tokenizers import CustomPunktTokenizer
    tok = CustomPunktTokenizer(model_path=model_path, name="test_custom")
    sents = tok.sent_tokenize("Esta es una frase. Otra frase aquí.")
    assert len(sents) == 2
    assert tok.name == "test_custom"
