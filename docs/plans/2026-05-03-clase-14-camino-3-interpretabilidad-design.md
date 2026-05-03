# Clase 14 — Camino 3 (Interpretabilidad mecanicista) Design

**Fecha:** 2026-05-03
**Autor:** Roberto Araneda + Claude
**Estado:** Aprobado, listo para implementation plan

---

## 1. Objetivo y filosofia

Pasar de "los modelos funcionan" (Caminos 1, 2, 2.5, 4) a **"entiendo que hacen por dentro"**. Construir herramientas de interpretabilidad mecanicista desde cero (forward hooks, cache de activaciones, decomposition matematica, activation patching, sparse autoencoders) y aplicarlas a los modelos ya entrenados del curso.

**Principio rector:** mantener la pedagogia "build it yourself" del curso. No usar TransformerLens ni librerias de interpretabilidad. Cada tecnica se implementa con primitivas de PyTorch (~10-50 lineas).

**Resultado esperado:** un lector que termine Camino 3 sabe:
- Como inspeccionar cualquier capa de un Transformer
- Que es el residual stream y como cada bloque suma a el
- Como identificar previous-token e induction heads
- La diferencia entre correlacion y causalidad en interpretabilidad
- Como entrenar e interpretar un sparse autoencoder
- Como cambia la interpretabilidad entre encoder-only (BERT) y decoder-only (LLaMA)

---

## 2. Modelos analizados

- **Mini-LLaMA char-level** (cap 21, vocab=65): protagonista. Visualizaciones limpias (heatmaps de 65 cols). Caps 50-61.
- **Mini-LLaMA SFT** (cap 24): para circuit discovery sobre tareas controladas (cap 58).
- **Mini-BERT fine-tuneado** (cap 47): contraste encoder vs decoder al final (cap 62).

**No se entrenan modelos base nuevos.** Solo se agrega un Sparse Autoencoder (caps 60-61) sobre activaciones cacheadas.

---

## 3. Secuencia de capitulos (14 caps, 50-63)

### Fase 12 — Hooks y residual stream (fundacional)

| Cap | Titulo | Tecnica |
|---|---|---|
| 50 | Forward hooks | `register_forward_hook`, cache de activaciones |
| 51 | Residual stream | Visualizacion del bus principal del Transformer |
| 52 | Logit lens | Proyectar residual intermedio al vocab via `lm_head` |

### Fase 13 — Atencion por dentro

| Cap | Titulo | Tecnica |
|---|---|---|
| 53 | Heatmaps de atencion | Captura `attn_weights` por capa/head |
| 54 | Previous-token heads | Score = atencion media a posicion t-1 |
| 55 | Induction heads | Patron `[A][B] ... [A] -> [B]`, demo numerica |
| 56 | QK / OV decomposition | $W_Q^T W_K$ y $W_V W_O$ por cabeza |

### Fase 14 — Causalidad e intervencion

| Cap | Titulo | Tecnica |
|---|---|---|
| 57 | Activation patching | Clean vs corrupted run, swap activations, medir efecto |
| 58 | Mini-circuit discovery | Encontrar circuito "repeat" del SFT cap 24 via patching |

### Fase 15 — Frontera moderna (SAEs)

| Cap | Titulo | Tecnica |
|---|---|---|
| 59 | Superposition + monosemantic features | Toy model, demo de polisemanticidad |
| 60 | Entrenar un SAE | d_model=128 -> d_features=512 con L1 sparsity |
| 61 | Interpretar features del SAE | Top-k tokens por feature, identificar features humanas |

### Fase 16 — Contraste BERT y cierre

| Cap | Titulo | Tecnica |
|---|---|---|
| 62 | Interpretabilidad en Mini-BERT | [CLS] aggregation, [SEP] pooling, sin induction heads |
| 63 | Comparativa final + frontera 2026 | Tabla maestra, links a Anthropic Circuits Thread, papers |

---

## 4. Infraestructura nueva

### Modulo `_interp.py`

Helpers compartidos por todos los scripts del camino:

```python
def cache_activations(model, names: list[str]) -> dict
    # Context manager. Registra hooks por nombre, retorna {name: tensor}.
    # Cleanup automatico al salir.

def get_attention_patterns(model, ids) -> Tensor  # (n_layers, n_heads, T, T)
    # Captura attn_weights de todas las capas en un forward pass.

def logit_lens(model, residual: Tensor) -> Tensor  # (B, T, vocab)
    # Aplica lm_head (sin LayerNorm final si aplica) al residual.

def patch_activation(model, ids, patch_dict) -> Tensor
    # Forward con activaciones reemplazadas.
    # patch_dict: {name: (positions, replacement_tensor)}

def qk_circuit(W_Q, W_K) -> Tensor  # (d_model, d_model)
def ov_circuit(W_V, W_O) -> Tensor  # (d_model, d_model)

def previous_token_score(attn_weights) -> float
    # Media de attn[i, i-1] para i > 0 — score "copia el anterior"

def induction_score(attn_weights, ids) -> float
    # Score sobre prompts repetidos [A][B]...[A]: cuanto atiende a la B previa
```

### SparseAutoencoder en `_interp.py` o `_sae.py`

```python
class SparseAutoencoder(nn.Module):
    # encoder: Linear(d_model, d_features) + ReLU
    # decoder: Linear(d_features, d_model)
    # loss = MSE(reconstruction) + lambda * L1(features)
```

### Tests nuevos en `tests/test_interp.py` (>=6)

- `test_cache_activations_captures_correct_shapes`
- `test_cache_activations_cleanup_removes_hooks`
- `test_logit_lens_consistent_with_full_forward` (logit lens del residual final == forward output)
- `test_patch_activation_changes_only_target_position`
- `test_qk_circuit_shape`
- `test_ov_circuit_shape`
- `test_sae_reconstruction_loss_decreases`

Total tests al final del Camino 3: 23 (Camino 4) + >=7 = >=30.

### Datasets nuevos

Ninguno versionado. Se cachean activaciones en memoria al vuelo desde corpus existentes (`shakespeare.txt`).

### Checkpoints nuevos

- `checkpoints/sae_mini_llama.pt` (gitignored, regenerable). ~1 MB.

---

## 5. Verificacion

### Por capitulo

1. **Helpers en `_interp.py`**: TDD pytest, todos los tests pasando.
2. **Scripts ejecutables**: corren sin error, output en rango esperado.
3. **Capitulos Hugo**: output literal incluido, `hugo --quiet` limpio.

### Criterios de exito por cap clave

| Cap | Criterio cuantitativo |
|---|---|
| 50 | Cache de 5 puntos (embed + 4 blocks) con shapes correctos |
| 54 | >=1 cabeza con previous-token score >0.5 (ajustable a la realidad observada) |
| 55 | >=1 cabeza con induction score >0.3 sobre prompts `[A][B]...[A]` |
| 57 | Patching de atencion en cabeza relevante restaura prediccion en >40% del efecto |
| 58 | Identificar >=2 cabezas necesarias para tarea "repeat" del SFT |
| 60 | SAE loss baja >=1 orden de magnitud (ej: 0.5 -> 0.05) |
| 61 | >=3 features con interpretacion humana plausible (ej: feature de mayusculas) |

### Honestidad pedagogica

Si un patron clasico **no emerge** a la escala de Mini-LLaMA (4 capas, d_model=128), se documenta honestamente en el cap correspondiente. Coherente con cap 29 (DPO regression) y cap 37 (BPE tradeoffs).

---

## 6. Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|---|---|---|
| Induction heads no emergen claras | Media | Documentar honestamente, mostrar las mas cercanas, discutir limitacion de escala |
| SAE no produce features monosemanticas | Media | Usar lambda y arquitectura de Bricken et al. 2023 como referencia. Si fallan, documentar polisemanticidad observada |
| Activation patching requiere prompts pareados | Baja | Cap 57: prompts Shakespeare-style (`BRUTUS:` vs `ROMEO:`). Cap 58: tareas SFT controladas |
| Mini-LLaMA tiene LayerNorm post-block | Baja | Logit lens debe aplicar LayerNorm antes de lm_head si el modelo lo hace en forward |

---

## 7. Conexiones con caminos previos

- **Cap 21 (Mini-LLaMA)**: modelo base analizado en caps 50-61
- **Cap 24 (SFT)**: modelo usado para circuit discovery (cap 58)
- **Cap 47 (Mini-BERT fine-tuned)**: modelo de contraste (cap 62)
- **Cap 41 (BERTBlock)**: estructura interna que se inspecciona en cap 62
- **Caps 17-19 (SwiGLU/RoPE/GQA)**: las modernizaciones LLaMA influyen en como se inspecciona cada componente

---

## 8. Bibliografia base

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage et al., 2021
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — Olsson et al., 2022
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Elhage et al., 2022
- [Interpretability in the Wild: IOI in GPT-2 small](https://arxiv.org/abs/2211.00593) — Wang et al., 2022
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Bricken et al., 2023
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Templeton et al., 2024
- [TransformerLens library](https://github.com/TransformerLensOrg/TransformerLens) — Neel Nanda (mencionado en cap 63 como herramienta profesional, no usado en el curso)

---

## 9. Outputs esperados al cierre del Camino 3

- 14 capitulos Hugo (`50-forward-hooks.md` ... `63-comparativa-interp-frontera.md`)
- 14 scripts ejecutables (`50_forward_hooks.py` ... `63_*.py` — el ultimo puede ser solo Hugo)
- 1 modulo nuevo: `_interp.py`
- 1 clase nueva: `SparseAutoencoder`
- >=7 tests nuevos en `tests/test_interp.py`
- 1 checkpoint nuevo gitignored: `sae_mini_llama.pt`
- 1 entrada glosario nueva: `site/content/fundamentos/interpretabilidad-mecanicista.md` (~1500 palabras) — actualizar la entrada `interpretabilidad.md` existente o crear nueva especifica
- Hub `_index.md` actualizado con Fase 12-16
- ~14-18 commits en branch `feat/clase-14-camino-3-interpretabilidad`

---

## 10. Estimacion de esfuerzo

Comparado con Camino 4 (12 caps + 11 scripts, ~18 tasks):
- Camino 3: 14 caps + 13 scripts + 1 modulo + 1 SAE training. Estimado **~20-22 tasks** en el implementation plan.
- Tiempo de training extra: SAE ~1-2 minutos sobre activaciones cacheadas (no es training de modelo grande).
- Tiempo total: similar a Camino 4 (~1-2 sesiones largas con subagent-driven-development).
