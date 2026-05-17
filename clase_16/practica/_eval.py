"""Métricas de evaluación: precision/recall/F1 sobre Entity y oraciones."""
from typing import Dict, List

from _corpora import Entity


def precision_recall_f1(predicted: List[Entity], gold: List[Entity],
                        match_mode: str = "exact") -> Dict[str, float]:
    """Calcula P, R, F1 entre lista de entidades predichas y gold.

    match_mode:
      - "exact": misma (start, end, label)
      - "partial": overlap > 0 con mismo label
      - "type_only": solo cuenta el label
    """
    if match_mode not in {"exact", "partial", "type_only"}:
        raise ValueError(f"unknown match_mode: {match_mode!r}")

    def matches(p: Entity, g: Entity) -> bool:
        if p.label != g.label:
            return False
        if match_mode == "exact":
            return p.start == g.start and p.end == g.end
        if match_mode == "partial":
            return not (p.end <= g.start or g.end <= p.start)
        return True  # type_only

    tp = sum(1 for p in predicted if any(matches(p, g) for g in gold))
    fp = len(predicted) - tp
    matched_gold = sum(1 for g in gold if any(matches(p, g) for p in predicted))
    fn = len(gold) - matched_gold

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def sentence_boundary_f1(predicted: List[str], gold: List[str]) -> float:
    """F1 set-based entre dos listas de oraciones."""
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
