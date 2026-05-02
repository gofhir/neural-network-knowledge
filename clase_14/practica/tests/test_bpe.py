"""Tests para _bpe.py — BPETokenizer y CharTokenizer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bpe import BPETokenizer

def test_bpe_train_reduces_token_count():
    """Entrenar con merges reduce tokens vs char-level."""
    corpus = "aaabdaaabac"
    tok = BPETokenizer()
    tok.train(corpus, num_merges=3)
    # despues de merges, vocab debe tener mas tokens que chars unicos
    assert len(tok.vocab) > len(set(corpus))
    # y al menos 1 merge fue registrado
    assert len(tok.merges) > 0

def test_bpe_round_trip():
    """encode → decode reproduce el texto original."""
    import os
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    corpus = open(corpus_path).read()
    tok = BPETokenizer()
    tok.train(corpus, num_merges=100)
    sample = "To be or not to be"
    ids = tok.encode(sample)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == sample

def test_bpe_encode_shorter_than_chars():
    """Con merges suficientes, encode produce menos tokens que chars."""
    import os
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    corpus = open(corpus_path).read()
    tok = BPETokenizer()
    tok.train(corpus, num_merges=500)
    sample = "the king is dead"
    ids = tok.encode(sample)
    assert len(ids) <= len(sample)
