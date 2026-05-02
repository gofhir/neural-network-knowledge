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
        raise NotImplementedError("implement in Task 2")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("implement in Task 2")

    def save(self, path: str) -> None:
        raise NotImplementedError("implement in Task 3")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        raise NotImplementedError("implement in Task 3")
