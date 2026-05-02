# Clase 14 — Camino 2: SFT + DPO sobre Mini-LLaMA

**Fecha:** 2026-05-01
**Estado:** diseño aprobado, pendiente plan de implementación
**Contexto previo:** Camino 1 cerrado en 2026-05-01 (Mini-GPT + Mini-LLaMA char-level entrenados desde cero, 21 capítulos Hugo + 17 scripts).

## Objetivo

Convertir el Mini-LLaMA pretrained de Camino 1 (que solo predice tokens char-level estilo Shakespeare) en un asistente que sigue instrucciones, vía el stack moderno **SFT → DPO**. RLHF/PPO queda fuera del alcance (la industria pasó a DPO).

El viaje es **pedagógico**, no productivo: el resultado final es un Mini-LLaMA que aprende a responder a 4 tareas sintéticas con formato `INSTR/RESP`, demostrando los procedimientos de loss y entrenamiento de SFT y DPO sin ruido de tokenización.

## Decisiones de alcance

| Dimensión | Decisión | Por qué |
|---|---|---|
| Etapas | SFT + DPO (no RLHF/PPO) | Stack moderno usado en Llama-3-Instruct, Mistral, Zephyr |
| Tokenización | Char-level (vocab 65 Shakespeare) | Aísla SFT/DPO del ruido de BPE; reusa infra de Camino 1 |
| Base model | Mini-LLaMA pretrained de Camino 1 | Es el sentido real de "fine-tuning"; no entrenamos desde cero |
| Tareas SFT | reverse, upper, repeat, Q&A factoide | Variedad sin extender vocab; métricas exact-match objetivas |
| Preferencias DPO | Mix base-sampled + cross-task | Captura tanto base-drift como instruction-following |
| Estructura | 8 capítulos en Fase 6 (SFT, caps 22-25) + Fase 7 (DPO, caps 26-29) | Sigue patrón granular de Fase 5 (LLaMA modernizations) |

## Arquitectura

### Scripts ejecutables — `clase_14/practica/`

```
14_show_base_no_instructions.py    cap 22 — demo del problema
15_build_sft_dataset.py            cap 23 — genera data/sft_dataset.jsonl
16_train_sft.py                    cap 24 — SFT loop con loss masking
17_eval_sft.py                     cap 25 — exact-match harness
18_dpo_intro.py                    cap 26 — Bradley-Terry, demo numérica
19_dpo_loss_derivation.py          cap 27 — loss DPO paso a paso
20_build_dpo_dataset.py            cap 28 — chosen/rejected (base+cross)
21_train_dpo.py                    cap 29 — DPO training + eval comparativa
```

### Capítulos Hugo — `site/content/clases/clase-14/practica/`

```
22-base-model-no-instructions.md    Fase 6 inicio
23-dataset-sft.md
24-sft-training.md
25-sft-eval.md
26-preferencias-bradley-terry.md    Fase 7 inicio
27-dpo-loss.md
28-dataset-dpo.md
29-dpo-training-eval.md
```

### Extensiones a `_models.py`

Nuevas funciones (no tocar las clases existentes):

- `load_pretrained_mini_llama(checkpoint_path, device) -> MiniLLaMA` — helper limpio
- `generate_with_prompt(model, prompt_str, char_to_id, id_to_char, max_new_tokens, temperature, top_k) -> str` — generación condicionada con encode/decode
- `compute_logp_response(model, prompt_ids, response_ids) -> torch.Tensor` — suma de log-probs por token de la respuesta (con shift correcto). Usado por DPO y por eval.
- `dpo_loss(policy, ref, prompt_ids, chosen_ids, rejected_ids, beta) -> torch.Tensor` — loss de un batch DPO

### Datasets versionados — `clase_14/practica/data/`

```
data/sft_dataset.jsonl       ~5000 pares
  formato: {"prompt": "...", "response": "...\n", "task": "reverse|upper|repeat|qa"}
  splits: 4000 train + 1000 eval (separados por seed de generación)

data/dpo_dataset.jsonl       ~3000 triples
  formato: {"prompt": "...", "chosen": "...", "rejected": "...", "source": "base|cross"}
  composición: 1500 base-sampled + 1500 cross-task
```

Ambos `.jsonl` se commitean al repo (deterministas con seeds 42/43, reproducibles, ~1MB total).

### Checkpoints — gitignored

```
checkpoints/mini_llama_base.pt    de Camino 1 (regenerable con 13_mini_llama.py)
checkpoints/mini_llama_sft.pt     output del cap 24
checkpoints/mini_llama_dpo.pt     output del cap 29
```

Los `.pt` no van al repo; cada script documenta cómo regenerarlos.

## Fase 6 — SFT (caps 22-25)

### Cap 22 — el problema

Cargar Mini-LLaMA base + dar prompts `INSTR: reverse 'cat'\nRESP: ` + observar que genera Shakespeare-ish. Output literal en el cap (verificado tras ejecutar el script). Setea la motivación de fine-tunear.

### Cap 23 — dataset SFT

4 tareas con generadores deterministas. Distribución pensada para evitar overfitting trivial:

| Tarea | Plantilla prompt | Generador | n |
|---|---|---|---|
| reverse | `INSTR: reverse 'WORD'\nRESP: ` | `WORD[::-1]` | 1500 |
| upper | `INSTR: upper 'WORD'\nRESP: ` | `WORD.upper()` | 1500 |
| repeat | `INSTR: repeat 'X' N\nRESP: ` | `X * N`, N∈{2,3,4} | 1500 |
| qa | `Q: PREGUNTA?\nA: ` | tabla fija de ~30 facts curados | 500 |

Q&A es chico porque es memorización pura. Más pares solo amplifican el overfit sin aportar señal.

**Filtro de vocab**: validar que cada `prompt+response` use solo chars del Shakespeare vocab (65 tokens). Ejemplos que no pasan se descartan. Si los facts de Q&A tienen chars fuera de vocab, se reformulan o descartan. El script imprime cuántos ejemplos se filtraron por tarea.

**Splits**: train (4000) y eval (1000) generados con seeds distintas para que los WORDs/Xs sean disjuntos. Q&A no tiene holdout (es memorización deliberada) — se documenta honestamente.

### Cap 24 — SFT loop

Lo distintivo vs pretrain de Camino 1:

- **Carga del base**: `model.load_state_dict(torch.load("checkpoints/mini_llama_base.pt", map_location=device))`. Si no existe, el script imprime un mensaje claro: `"Run 13_mini_llama.py first to produce mini_llama_base.pt"`.
- **Loss masking en prompt**: `loss_mask` binario de la longitud del input; tokens del prompt → 0, tokens de response → 1. Multiplicamos antes de promediar. **Crítico** y poco intuitivo — el cap entero lo explica con diagrama.
- **Hyperparams** (justificados en el cap):
  - `lr = 1e-4` (10× menor que pretrain — convención SFT)
  - `max_iters = 1500`
  - `batch_size = 32`
  - `block_size = 64` (mismo que base)
  - sin warmup (dataset chico)
  - `weight_decay = 0.01`
- **Output**: `checkpoints/mini_llama_sft.pt` + log de loss por iter.

### Cap 25 — eval SFT

Harness con `_eval.py` (módulo nuevo compartido):

```python
def eval_exact_match(model, dataset_jsonl, n=200, max_new_tokens=20) -> dict[task, acc]
def eval_qualitative(model, prompts, n_samples=3, temperature=0.8) -> dict[prompt, list[str]]
def eval_drift(model, ambiguous_prompts) -> float  # n-gram heuristic Shakespeare-style
```

**Tabla esperada en el cap** (los valores reales se llenan tras ejecutar):

| Tarea | Base | SFT |
|---|---|---|
| reverse_acc | ~0% | 80-95% |
| upper_acc | ~0% | 85-95% |
| repeat_acc | ~0% | 90-99% |
| qa_acc | ~0% | 70-90% |

Más eval **cualitativo**: 5 ejemplos de generación literal por tarea, mostrando el "salto" de Shakespeare-ish → instrucciones.

## Fase 7 — DPO (caps 26-29)

### Cap 26 — preferencias y Bradley-Terry

Sin código pesado; construye intuición:

- Por qué SFT no es suficiente: enseña formato pero no "qué respuesta es mejor entre dos válidas".
- Bradley-Terry: dado `(y_w, y_l)`, modelar `P(y_w ≻ y_l) = σ(r(x,y_w) - r(x,y_l))` con un reward implícito.
- Demo numérica chica en `18_dpo_intro.py`: dos respuestas a una pregunta, mostrar cómo la log-likelihood de la preferencia depende de los rewards.
- Cierre: "RLHF aprende r explícito + PPO; DPO se salta el reward y va directo a la policy".

### Cap 27 — derivación de la loss DPO

Paso a paso, matemáticamente:

```
L_DPO = -E[(x,y_w,y_l)] log σ( β · log π_θ(y_w|x)/π_ref(y_w|x)
                              - β · log π_θ(y_l|x)/π_ref(y_l|x) )
```

- `π_θ` = policy (modelo a entrenar, init = SFT checkpoint)
- `π_ref` = reference (SFT congelado)
- Por qué los **log-ratios**: KL implícito al ref evita colapso de la policy.
- Qué hace `β`: 0.1 conservador, 1.0 agresivo.
- Implementación numérica: `log π(y|x)` = suma de log-probs por token sobre los tokens de respuesta (con loss_mask, mismo principio que SFT).
- Mini-demo en `19_dpo_loss_derivation.py`: calcular la loss para un único triple a mano y verificar coherencia.

### Cap 28 — dataset DPO (mix base + cross-task)

```
total: 3000 triples
├─ 1500 base-sampled:
│    para cada prompt del SFT dataset, sampleamos del base model pre-SFT
│    chosen   = respuesta correcta del SFT dataset
│    rejected = lo que generó el base model (Shakespeare drift)
│
└─ 1500 cross-task:
     chosen   = respuesta correcta para tarea A
     rejected = respuesta de tarea B (B≠A) sobre el mismo input
     ej: prompt "INSTR: reverse 'cat'", chosen "tac", rejected "CAT"
```

**Filtros**:
- Descartar triples donde `chosen == rejected`.
- Validar vocab (mismo filtro que SFT).
- El script `20_build_dpo_dataset.py` imprime cuántos triples se filtraron y la composición final por `source`.

**Formato JSONL**:
```json
{"prompt": "...", "chosen": "...", "rejected": "...", "source": "base|cross"}
```

### Cap 29 — DPO training + eval comparativa final

- Cargamos **dos** modelos: `policy = SFT_checkpoint.copy()`, `ref = SFT_checkpoint` (frozen, `requires_grad=False`).
- Loop: por batch de triples, calcular `log π(y_w)`, `log π(y_l)`, `log π_ref(y_w)`, `log π_ref(y_l)` → loss DPO.
- **Hyperparams**:
  - `lr = 5e-5`
  - `β = 0.1`
  - `max_iters = 1000`
  - `batch_size = 16` (más memoria por dos forward passes simultáneos)
- **Eval final** — tabla de 3 columnas:

| Tarea | Base (Camino 1) | SFT | SFT+DPO |
|---|---|---|---|
| reverse_acc | ~0% | 80-95% | igual o ligeramente mejor |
| upper_acc | ~0% | 85-95% | igual o ligeramente mejor |
| repeat_acc | ~0% | 90-99% | igual o ligeramente mejor |
| qa_acc | ~0% | 70-90% | igual o ligeramente mejor |
| drift_score (cualitativo) | alto | medio | bajo |

**Análisis honesto**: char-level + tareas determinísticas hace que SFT ya sature. DPO mejora poco en exact-match — donde brilla es en samples cualitativos sobre prompts ambiguos/OOD: no decae a Shakespeare cuando el prompt no es exacto. El cap incluye un set de **prompts ambiguos** (typos, formato cercano pero distinto) donde DPO sí muestra ventaja medible.

**Decisión técnica**: `ref_model` y `policy` ambos en MPS. Para vocab 65 + 4 capas el costo de memoria es trivial (~10 MB por modelo).

## Hiperparámetros consolidados

| Param | Pretrain (Camino 1) | SFT (cap 24) | DPO (cap 29) |
|---|---|---|---|
| `vocab_size` | 65 | 65 | 65 |
| `block_size` | 64 | 64 | 64 |
| `batch_size` | 32 | 32 | 16 |
| `d_model` | 128 | 128 | 128 |
| `h_q / h_kv` | 4 / 2 | 4 / 2 | 4 / 2 |
| `n_layers` | 4 | 4 | 4 |
| `d_ff` | 384 (SwiGLU) | 384 | 384 |
| `learning_rate` | 3e-4 | 1e-4 | 5e-5 |
| `max_iters` | 3000 | 1500 | 1000 |
| `weight_decay` | 0.01 | 0.01 | 0.01 |
| `β (DPO)` | — | — | 0.1 |
| `device` | mps | mps | mps |

Tiempos estimados en MPS (Apple Silicon):
- Pretrain (ya hecho): 25-40 s
- SFT: 15-20 s
- DPO: 30-40 s (dos forward passes por step)

## Reproducibilidad

- `torch.manual_seed(1337)` + `random.seed(1337)` en cada script de training.
- `SFT_SEED = 42`, `DPO_SEED = 43` para generación de datasets (deterministas).
- Datasets `.jsonl` versionados al repo.
- Checkpoints `.pt` gitignored, regenerables.

## Glosario nuevo — `site/content/fundamentos/`

Términos a sumar al glosario existente del curso:

- `sft.md` — Supervised Fine-Tuning
- `dpo.md` — Direct Preference Optimization
- `bradley-terry.md` — modelo de preferencias
- `kl-implicito.md` — KL implícito vs explícito
- `loss-masking.md` — masking en SFT

(Confirmado integrar al glosario como parte del Camino 2.)

## Update del hub

`site/content/clases/clase-14/practica/_index.md`: agregar **Fase 6 (SFT)** y **Fase 7 (DPO)** con cards a los caps 22-29, manteniendo el patrón de las Fases 1-5.

## Riesgos identificados y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Modelo sobreajusta a 5000 ejemplos sintéticos | Holdout train/eval con seeds disjuntas; reportar accuracy honesta sobre eval |
| SFT satura métricas → DPO se ve trivial | Eval cualitativo con prompts OOD/ambiguos donde DPO sí muestra ventaja |
| Q&A factoide es memorización pura | Documentado explícitamente; n=500 limita el peso |
| Char nuevo en synthetic → vocab mismatch | Filtro de vocab en build dataset; printeamos descartes |
| Confusión sobre por qué cargar dos modelos en DPO | Cap 27 dedica diagrama explícito a `policy vs ref` |
| Loss masking en SFT no se entiende | Cap 24 tiene una sub-sección con diagrama de tokens enmascarados |

## Convenciones del estilo (heredadas de Caminos 1)

- Pedagogía conversacional: explicación → script → output → preguntas de verificación.
- Capítulos Hugo en español **sin tildes** en filenames (`dataset-sft.md`, no `dataset-sft.md`).
- Output literal de scripts incluido en los caps (verificado, no inventado).
- Comparación con escalones previos: cada nueva pieza vs Mini-LLaMA pretrained.
- Scripts numerados con prefijos consistentes (continúan de 13 → 14, 15, ...).

## Estado al cierre del diseño

- Diseño aprobado sección por sección por el usuario.
- Plan de implementación pendiente (siguiente paso: invocar `superpowers:writing-plans`).
- Memoria del proyecto (`project_clase_14_caminos_pendientes.md`) se actualizará al terminar el Camino 2.
