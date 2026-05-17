"""Wrappers de tokenizadores NLTK con interfaz uniforme."""
import pickle
from pathlib import Path
from typing import Dict, List, Protocol

from nltk.tokenize import (
    TweetTokenizer as _NLTKTweet,
    sent_tokenize,
    word_tokenize,
)
from nltk.tokenize.punkt import PunktSentenceTokenizer


class Tokenizer(Protocol):
    """Protocolo común a todos los tokenizadores."""
    name: str

    def tokenize(self, text: str) -> List[str]: ...
    def sent_tokenize(self, text: str) -> List[str]: ...


class NLTKPunktTokenizer:
    """Wrapper sobre NLTK Punkt sentence + Treebank word tokenizer."""

    def __init__(self, language: str = "spanish"):
        self.language = language
        self.name = f"punkt_{language[:2]}"

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text, language=self.language)

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text, language=self.language)


class NLTKTreebankTokenizer:
    """Treebank/Penn word tokenizer estándar de NLTK (default inglés)."""
    name = "treebank"

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)


class TweetTokenizer:
    """TweetTokenizer de NLTK; preserva emoticonos, hashtags y menciones."""
    name = "tweet"

    def __init__(self):
        self._tk = _NLTKTweet()

    def sent_tokenize(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return self._tk.tokenize(text)


class CustomPunktTokenizer:
    """Tokenizer Punkt entrenado en un sub-corpus específico.

    Carga parámetros (PunktParameters) pickled desde disco e inicializa
    un PunktSentenceTokenizer con ellos.
    """

    def __init__(self, model_path: Path, name: str = "punkt_custom"):
        self.name = name
        self.model_path = Path(model_path)
        with open(self.model_path, "rb") as f:
            params = pickle.load(f)
        self._tk = PunktSentenceTokenizer()
        self._tk._params = params

    def sent_tokenize(self, text: str) -> List[str]:
        return self._tk.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)


def list_tokenizers() -> Dict[str, Tokenizer]:
    """Diccionario de tokenizadores instanciados, listos para usar."""
    return {
        "punkt_es": NLTKPunktTokenizer(language="spanish"),
        "punkt_en": NLTKPunktTokenizer(language="english"),
        "treebank": NLTKTreebankTokenizer(),
        "tweet": TweetTokenizer(),
    }
