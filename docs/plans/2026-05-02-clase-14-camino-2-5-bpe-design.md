# Clase 14 — Camino 2.5: BPE + SFT + DPO Design

**Fecha:** 2026-05-02
**Estado:** diseño aprobado, pendiente plan de implementación
**Contexto previo:** Camino 2 cerrado (caps 22-29, char-level SFT+DPO). Limitación detectada: char-level con d_model=128 tiene accuracy techo bajo (21-23% en reverse/upper), impidiendo demostrar DPO limpiamente.

## Motivación

Camino 2 demostró los mecanismos de SFT y DPO pero no pudo mostrar wins claros de DPO sobre SFT porque:
- SFT char-level ya saturaba el techo arquitectónico en tareas difíciles (reverse/upper ~21%)
- DPO con β=0.1 over-optimizó y degradó accuracy en todas las tareas
- Char-level sin semántica de palabras limita la calidad de los rejected en DPO

Camino 2.5 resuelve esto con **BPE tokenización** (~1000 merges, vocab ~1000 tokens), un nuevo pretrain, y tareas BPE-naturales. Coexiste con Camino 2 como addendum optativo — el contraste char vs BPE ES la lección.

## Decisiones de diseño

| Dimensión | Decisión | Alternativas rechazadas |
|---|---|---|
| Posición | Camino 2.5 (caps 30-37), addendum de Camino 2 | Reemplazar C2 (pierde lección honesta), nuevo Camino (renumera pendientes) |
| Tokenizer | BPE desde cero, ~1000 merges | tiktoken 50k (desbalancea modelo), SentencePiece (lib externa) |
| Tareas | Híbrido: qa + repeat + complete-en + complete-es | Solo char tasks (anti-natural para BPE), solo nuevas (pierde comparación) |
| Infra | Refactor tokenizer-agnostic (CharTokenizer wrapper) | Módulos paralelos (duplicación innecesaria), todo separado (no reutiliza DPO math) |
| Beta DPO | β=0.1 primero + sweep β=0.5 | Solo β=0.1 (no valida hipótesis cap 29) |

## Arquitectura

### Branch y files

Branch: `feat/clase-14-camino-2.5-bpe` desde HEAD de `feat/clase-14-camino-2-sft-dpo`.

```
clase_14/practica/
  _bpe.py                     NUEVO: BPETokenizer class + CharTokenizer wrapper
  _models.py                  REFACTOR: generate_with_prompt acepta tokenizer object
  _eval.py                    REFACTOR: eval functions tokenizer-agnostic

  30_build_bpe.py             cap 30: entrenar BPETokenizer sobre Shakespeare+Quijote
  31_pretrain_bpe.py          cap 31: pretrain Mini-LLaMA vocab=1000
  32_tokenizer_refactor_demo.py  cap 32: demo que refactor no rompe nada
  33_build_sft_bpe.py         cap 33: dataset 4 tareas BPE (4000 train + 1000 eval)
  34_train_sft_bpe.py         cap 34: SFT con BPE + eval comparativo
  35_build_dpo_bpe.py         cap 35: dataset DPO-BPE (3000 triples)
  36_train_dpo_bpe.py         cap 36: DPO + beta sweep + eval final
  37_compare_char_vs_bpe.py   cap 37: tabla maestra char vs BPE

  data/
    bpe_tokenizer.json        versionado (vocab + merges)
    sft_bpe_dataset.jsonl     versionado (~4000 pares)
    sft_bpe_eval.jsonl        versionado (~1000 pares)
    dpo_bpe_dataset.jsonl     versionado (~3000 triples)

  tests/
    test_bpe.py               NUEVO: 4 tests BPETokenizer
    test_models_helpers.py    ACTUALIZAR: verificar compatibilidad post-refactor

  checkpoints/
    mini_llama_bpe_base.pt    gitignored
    mini_llama_bpe_sft.pt     gitignored
    mini_llama_bpe_dpo.pt     gitignored

site/content/clases/clase-14/practica/
  30-bpe-desde-cero.md … 37-comparacion-char-vs-bpe.md

site/content/fundamentos/
  bpe.md                      NUEVO: ~1500 palabras
```

### `_bpe.py` — módulo BPETokenizer

```python
class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}       # token_str → id
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []  # orden de merges aprendidos

    def train(self, corpus: str, num_merges: int) -> None:
        # 1. vocab inicial = chars únicos del corpus
        # 2. tokenizar corpus como lista de chars
        # 3. num_merges veces: contar pares, elegir el más frecuente, mergear
        ...

    def encode(self, text: str) -> list[int]:
        # chars del texto → aplicar merges en orden → devolver IDs
        ...

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids)

    def save(self, path: str) -> None: ...  # JSON
    @classmethod
    def load(cls, path: str) -> "BPETokenizer": ...


class CharTokenizer:
    """Wrapper char-level que expone la misma interfaz que BPETokenizer.
    Permite que código char-level de Camino 2 funcione sin tocar nada.
    """
    def __init__(self, char_to_id: dict, id_to_char: dict):
        self.char_to_id = char_to_id
        self.id_to_token = id_to_char
        self.vocab_size = len(char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)
```

### Refactor de `_models.py` (minimal)

```python
# generate_with_prompt: char_to_id, id_to_char → tokenizer
def generate_with_prompt(model, prompt, tokenizer, max_new_tokens=50,
                         temperature=1.0, top_k=None, device=None,
                         stop_token="\n"):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device or get_device())
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        stop_ids = tokenizer.encode(stop_token) if stop_token else []
        if stop_ids and next_id.item() == stop_ids[0]:
            break
    return tokenizer.decode(x[0].tolist())
```

`compute_logp_response` y `dpo_loss` no cambian (ya operan sobre IDs).

### Refactor de `_eval.py` (minimal)

```python
def eval_exact_match(model, dataset_jsonl, tokenizer, n_per_task=200, ...):
    ...
    full = generate_with_prompt(model, ex["prompt"], tokenizer, ...)
    ...
```

### Actualización de scripts char-level (Camino 2) — 1 línea por script

```python
# Antes:
out = generate_with_prompt(model, prompt, c2i, i2c, ...)

# Después:
from _bpe import CharTokenizer
tokenizer = CharTokenizer(c2i, i2c)
out = generate_with_prompt(model, prompt, tokenizer, ...)
```

### Tests nuevos — `tests/test_bpe.py`

- `test_bpe_round_trip`: `encode → decode == texto original`
- `test_bpe_save_load`: guardar/cargar reproduce encode/decode idéntico
- `test_bpe_train_reduces_token_count`: con merges > 0, len(encode(texto)) < len(texto)
- `test_char_tokenizer_compat`: CharTokenizer produce mismo encode que dict directo

## Tareas SFT-BPE (Cap 33)

| Tarea | Plantilla prompt | Generador | n |
|---|---|---|---|
| qa | `Q: PREGUNTA?\nA: ` | ~40 facts bilingüe (EN+ES) | 1000 |
| repeat | `INSTR: repeat 'X' N\nRESP: ` | X char, N en {two,three,four} | 1000 |
| complete-en | `EN: 'FRASE'\nNEXT: ` | última palabra de línea Shakespeare | 1000 |
| complete-es | `ES: 'FRASE'\nNEXT: ` | última palabra de línea Quijote | 1000 |

**Extracción de complete-***: líneas del corpus con 20-60 chars, última palabra como target, resto como prompt. Filtros: target debe estar en vocab BPE (siempre que BPE entrenó en mismo corpus), prompt no puede terminar con signo de puntuación aislado.

**qa bilingüe (~40 facts)**:
- Facts EN ya definidos en Camino 2 (capitales, autores literarios)
- Facts ES nuevos: "¿quien escribio Don Quijote? → Cervantes", "¿cual es la capital de España? → Madrid", "¿quien escribio La Odisea? → Homero"

**Splits**: 4000 train + 1000 eval con seeds distintas (SFT_BPE_SEED=142, EVAL_BPE_SEED=1242).

## BPE training config

```
corpus:         shakespeare.txt + quijote.txt (bilingüe, ~1MB)
num_merges:     1000
vocab inicial:  ~100 chars únicos (corpus bilingüe tiene más que solo Shakespeare)
vocab final:    ~1100 tokens
output:         data/bpe_tokenizer.json
```

Con 1000 merges sobre 1MB de texto, esperamos vocab de palabras comunes ("the", "and", "que", "de") + subwords + chars raros. "house" debería ser 1-2 tokens. "Shakespeare" probablemente 3-4 tokens.

## Pretrain BPE config

```
vocab_size:    1100 (aprox, depende del training BPE)
max_seq_len:   256
d_model:       128
h_q / h_kv:   4 / 2
n_layers:      4
d_ff:          384 (SwiGLU)
lr:            3e-4
max_iters:     3000
batch_size:    32
device:        mps
corpus:        shakespeare.txt + quijote.txt tokenizado BPE
checkpoint:    checkpoints/mini_llama_bpe_base.pt
```

**Estimación embedding size**: 1100 × 128 = 140k params (vs 65 × 128 = 8k char-level). Modelo total ~1.1M params, embedding no domina.

## SFT-BPE config

```
lr:          1e-4
max_iters:   1500
batch_size:  32
tokenizer:   BPETokenizer.load("data/bpe_tokenizer.json")
base:        mini_llama_bpe_base.pt
output:      mini_llama_bpe_sft.pt
```

## DPO-BPE config

```
lr:          5e-5
beta:        0.1 (primero) + 0.5 (sweep en mismo script)
max_iters:   1000 por beta
batch_size:  16
dataset:     dpo_bpe_dataset.jsonl (3000 triples, 50% base-sampled / 50% cross-task)
base:        mini_llama_bpe_sft.pt (policy y ref)
output:      mini_llama_bpe_dpo.pt (del mejor beta)
```

## Eval methodology

**Comparación char vs BPE** (cap 37): cargar los 6 modelos (base/sft/dpo × char/bpe) y correr `eval_exact_match` sobre subset compartido (qa + repeat que existen en ambos). Mide aisladamente el efecto de tokenización.

**Eval cualitativo**: para `complete-en` y `complete-es`, mostrar 3-5 ejemplos literales del output del modelo BPE (base vs SFT vs DPO) — estas tareas son las nuevas y su output es legible para cualquier lector.

**Drift**: correr `eval_drift` sobre prompts ambiguos. Esperamos que BPE-base tenga drift similar a char-base (40%), BPE-SFT baje a ~0%, BPE-DPO también ~0%.

**Beta sweep doc**: el cap 36 incluye tabla:

| Beta | DPO loss final | qa acc | repeat acc | complete-en acc | complete-es acc |
|---|---|---|---|---|---|
| 0.1 | ? | ? | ? | ? | ? |
| 0.5 | ? | ? | ? | ? | ? |

Si β=0.5 es mejor, se documenta por qué (KL más restrictivo, menos over-optimization). Si ambos degradan, se propone iterar sobre dataset (limpiar cross-task).

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| stop_token encoding con BPE | `\n` podría no ser un token único en BPE — en ese caso usar EOS token explícito o longitud máxima como stop |
| complete-* demasiado fácil (SFT satura al 100%) | Hacer targets menos predecibles: usar palabras en posición -2 o -3 en vez de -1 |
| complete-* demasiado difícil (SFT < 30%) | Reducir ventana de extracton: líneas más cortas, targets más predecibles |
| qa bilingüe con chars fuera de vocab BPE | BPE entrenado en el corpus bilingüe debe cubrir todos los chars — verificar con filtro post-training |
| Refactor rompe scripts Camino 2 | pytest 6/6 antes y después del refactor como gate obligatorio |
| Beta sweep duplica tiempo de DPO | ~14 min total para 2 betas — aceptable |

## Capítulos Hugo (Fase 8)

```
30-bpe-desde-cero.md          Fase 8 inicio — algoritmo BPE, _bpe.py, ejemplos
31-pretrain-bpe.md             Pretrain bilingüe, perplexity vs char-level
32-refactor-tokenizer.md       Refactor agnostic, CharTokenizer wrapper, tests
33-dataset-sft-bpe.md          4 tareas BPE, complete-* construction
34-sft-bpe.md                  SFT-BPE + eval comparativo tabla 4 columnas
35-dataset-dpo-bpe.md          Dataset DPO-BPE, rejected más ricos con BPE
36-dpo-bpe.md                  DPO-BPE + beta sweep + análisis
37-comparacion-char-vs-bpe.md  Tabla maestra, lección tokenización
```

## Convenciones heredadas

Igual que Camino 2:
- Pedagogia conversacional: concepto → script → output literal → preguntas
- Scripts numerados `30_*.py` … `37_*.py` (matchean caps Hugo)
- Capítulos Hugo en español sin tildes
- Output real en caps (no inventado)
- Honestidad sobre resultados (si DPO falla de nuevo, documentar)

## Estado al cierre del diseño

- Diseño aprobado sección por sección por el usuario.
- Plan de implementación pendiente (siguiente paso: invocar `superpowers:writing-plans`).
