import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bpe import BPETokenizer


def test_bpe_special_tokens_extension():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    tok = BPETokenizer()
    tok.train(open(corpus_path).read(), num_merges=10)
    original_size = tok.vocab_size
    tok.add_special_tokens()
    assert tok.vocab_size == original_size + 3
    assert "[CLS]" in tok.vocab
    assert "[SEP]" in tok.vocab
    assert "[MASK]" in tok.vocab
    assert tok.cls_id == tok.vocab["[CLS]"]
    assert tok.sep_id == tok.vocab["[SEP]"]
    assert tok.mask_id == tok.vocab["[MASK]"]


def test_encode_bert_adds_cls_sep():
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "shakespeare.txt")
    tok = BPETokenizer()
    tok.train(open(corpus_path).read(), num_merges=10)
    tok.add_special_tokens()
    ids = tok.encode_bert("hello")
    assert ids[0] == tok.cls_id
    assert ids[-1] == tok.sep_id
    assert len(ids) >= 3
