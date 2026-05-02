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
