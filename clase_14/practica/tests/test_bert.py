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


def test_learned_pos_emb_shape():
    import torch
    from _models import LearnedPositionalEmbedding
    emb = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    x = torch.zeros(2, 10, 64)
    out = emb(x)
    assert out.shape == (2, 10, 64)


def test_learned_pos_emb_different_positions():
    import torch
    from _models import LearnedPositionalEmbedding
    emb = LearnedPositionalEmbedding(max_seq_len=128, d_model=64)
    x = torch.zeros(1, 5, 64)
    out = emb(x)
    assert not torch.all(out[0, 0] == out[0, 1])


def test_bert_block_shape():
    import torch
    from _models import BERTBlock
    block = BERTBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)


def test_bert_block_bidirectional():
    """El token 0 puede atender al token 9 (bidireccional, no causal)."""
    import torch
    from _models import BERTBlock
    torch.manual_seed(42)
    block = BERTBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(1, 10, 64)
    x1 = x.clone(); x2 = x.clone()
    x2[0, 9] = x2[0, 9] * 10
    out1 = block(x1); out2 = block(x2)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-4)
