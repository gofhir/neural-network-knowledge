"""_bpe.py — BPE tokenizer desde cero y CharTokenizer wrapper.
NOTE: clases de este archivo se usan en Camino 2.5 (caps 30-37).
"""
from __future__ import annotations
import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, corpus: str, num_merges: int) -> None:
        """Entrena BPE sobre corpus. Usa primeros 50k chars para velocidad pedagogica."""
        # Vocab inicial: chars unicos del corpus COMPLETO (no truncado)
        for c in sorted(set(corpus)):
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.id_to_token[idx] = c

        # Loop de merges sobre los primeros 50k chars (rapido, pedagogicamente suficiente)
        tokens = list(corpus[:50_000])

        for _ in range(num_merges):
            counts = Counter()
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += 1
            if not counts:
                break

            a, b = max(counts, key=counts.get)
            new_token = a + b
            self.merges.append((a, b))

            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.id_to_token[idx] = new_token

            # Aplicar merge
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        """Aplicar merges aprendidos en orden para tokenizar el texto."""
        tokens = [c for c in text if c in self.vocab]
        for a, b in self.merges:
            new_token = a + b
            if new_token not in self.vocab:
                continue
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def save(self, path: str) -> None:
        data = {"vocab": self.vocab, "merges": self.merges}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.id_to_token = {int(i): t for t, i in tok.vocab.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok


class CharTokenizer:
    """Wrapper char-level que expone la interfaz BPETokenizer."""
    def __init__(self, char_to_id: dict, id_to_char: dict):
        self._c2i = char_to_id
        self.id_to_token = id_to_char
        self.vocab_size = len(char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)
