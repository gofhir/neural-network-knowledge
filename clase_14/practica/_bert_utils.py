"""_bert_utils.py — utilidades para MLM masking y datasets BERT."""
import torch


def apply_mlm_mask(input_ids: torch.Tensor, mask_prob: float = 0.15,
                   mask_id: int = 1114, vocab_size: int = 1115,
                   special_ids: tuple = (1112, 1113, 1114)) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica masking MLM con split 80/10/10 de BERT.

    Returns:
        masked_ids: input_ids con tokens reemplazados
        labels:     original ids donde enmascarado, -100 donde no se predice
    """
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    B, T = input_ids.shape
    for b in range(B):
        for t in range(T):
            tok = input_ids[b, t].item()
            if tok in special_ids:
                continue
            if torch.rand(1).item() < mask_prob:
                labels[b, t] = tok
                r = torch.rand(1).item()
                if r < 0.80:
                    masked_ids[b, t] = mask_id
                elif r < 0.90:
                    masked_ids[b, t] = torch.randint(0, vocab_size - 3, (1,)).item()

    return masked_ids, labels
