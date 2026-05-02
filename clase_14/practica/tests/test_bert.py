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


def test_mini_bert_forward_shape():
    import torch
    from _models import MiniBERT
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    x = torch.randint(0, 1115, (2, 20))
    h = model(x)
    assert h.shape == (2, 20, 64)


def test_mlm_head_shape():
    import torch
    from _models import MiniBERT, MLMHead
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    head = MLMHead(d_model=64, vocab_size=1115)
    x = torch.randint(0, 1115, (2, 20))
    logits = head(model(x))
    assert logits.shape == (2, 20, 1115)


def test_classification_head_uses_cls():
    import torch
    from _models import MiniBERT, ClassificationHead
    model = MiniBERT(vocab_size=1115, max_seq_len=128, d_model=64,
                     n_heads=4, n_layers=2, d_ff=256)
    head = ClassificationHead(d_model=64, n_classes=2)
    x = torch.randint(0, 1115, (3, 20))
    logits = head(model(x))
    assert logits.shape == (3, 2)


def test_mlm_mask_proportion():
    import torch
    from _bert_utils import apply_mlm_mask
    ids = torch.randint(0, 1112, (1, 100))
    masked_ids, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                         mask_id=1114, vocab_size=1115)
    n_masked = (labels != -100).sum().item()
    assert 5 <= n_masked <= 30


def test_mlm_labels_minus100_for_unmasked():
    import torch
    from _bert_utils import apply_mlm_mask
    ids = torch.randint(0, 1112, (1, 50))
    _, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                mask_id=1114, vocab_size=1115)
    assert (labels >= 0).sum() <= 50
    assert (labels == -100).sum() + (labels >= 0).sum() == 50


def test_mlm_special_tokens_never_masked():
    """[CLS], [SEP], [MASK] nunca se enmascaran, incluso con mask_prob=1.0."""
    import torch
    from _bert_utils import apply_mlm_mask
    ids = torch.tensor([[1112, 100, 200, 1113]])
    _, labels = apply_mlm_mask(ids.clone(), mask_prob=1.0,
                                mask_id=1114, vocab_size=1115,
                                special_ids=(1112, 1113, 1114))
    assert labels[0, 0].item() == -100
    assert labels[0, -1].item() == -100
    assert labels[0, 1].item() != -100
    assert labels[0, 2].item() != -100
