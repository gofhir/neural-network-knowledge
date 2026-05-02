---
title: "37 - Comparacion final: char-level vs BPE-level — cierre Camino 2.5"
weight: 370
math: true
---

## 1. Apertura — el momento de la verdad

El Camino 2 (char-level) cerro en el [cap 29]({{< relref "29-dpo-training-eval" >}}) con un puzzle: SFT funcionaba, pero DPO degradaba. El Camino 2.5 (BPE-level) intento resolverlo trayendo todo al setup mas realista — vocabulario subword, dataset bilingue, rejected linguisticamente ricos. Aqui esta el veredicto.

Este capitulo es el centro de gravedad del Camino 2.5. Hasta ahora vimos los modelos uno a uno; ahora los ponemos los seis sobre la misma tabla — `char-base`, `char-sft`, `char-dpo`, `bpe-base`, `bpe-sft`, `bpe-dpo-b05` — evaluados sobre **el mismo subset compartido** (qa + repeat, las dos tareas que existen en ambos tokenizadores). La respuesta no es "BPE gana" ni "char-level gana": es **"depende, y aqui esta exactamente de que depende"**.

El punto pedagogico de este capitulo importa mas que las cifras: **cuando elegir cada tokenizacion**, y por que la respuesta nunca aparece tan limpiamente en los blogs y tutoriales que solo hablan de "scale up to BPE".

---

## 2. El script

`clase_14/practica/37_compare_char_vs_bpe.py`:

```python
"""37_compare_char_vs_bpe.py - Cap 37: tabla maestra char-level vs BPE-level.

Eval sobre el subset compartido (qa + repeat que existen en ambos tokenizadores).
"""
import torch, json
from pathlib import Path
from _bpe import BPETokenizer, CharTokenizer
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_drift, load_jsonl

device = "cpu"  # eval ligero — cpu es suficiente

# Char-level setup
text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
char_tok = CharTokenizer(c2i, i2c)
char_cfg = dict(vocab_size=len(c2i), max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# BPE-level setup
bpe_tok = BPETokenizer.load("data/bpe_tokenizer.json")
bpe_cfg = dict(vocab_size=bpe_tok.vocab_size, max_seq_len=256,
               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# Subset compartido: qa + repeat del eval set BPE
shared = [ex for ex in load_jsonl("data/sft_bpe_eval.jsonl")
          if ex["task"] in {"qa", "repeat"}]
with open("/tmp/shared_eval.jsonl", "w") as f:
    for ex in shared:
        f.write(json.dumps(ex) + "\n")
print(f"Shared eval examples: {len(shared)} (qa + repeat)")

ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]

print("\n=== Tabla maestra: char-level vs BPE-level ===\n")
print(f"{'modelo':<22}{'qa':<10}{'repeat':<10}{'drift':<10}")
print("-" * 52)

for label, ckpt, tok_obj, cfg in [
    ("char-base",   "checkpoints/mini_llama_base.pt",          char_tok, char_cfg),
    ("char-sft",    "checkpoints/mini_llama_sft.pt",           char_tok, char_cfg),
    ("char-dpo",    "checkpoints/mini_llama_dpo.pt",           char_tok, char_cfg),
    ("bpe-base",    "checkpoints/mini_llama_bpe_base.pt",      bpe_tok,  bpe_cfg),
    ("bpe-sft",     "checkpoints/mini_llama_bpe_sft.pt",       bpe_tok,  bpe_cfg),
    ("bpe-dpo-b05", "checkpoints/mini_llama_bpe_dpo_b05.pt",   bpe_tok,  bpe_cfg),
]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "/tmp/shared_eval.jsonl", tok_obj,
                          n_per_task=100, device=device)
    drift = eval_drift(m, ambiguous, tok_obj, device=device)
    qa  = em.get("qa", 0.0)
    rep = em.get("repeat", 0.0)
    print(f"{label:<22}{qa:<10.3f}{rep:<10.3f}{drift:<10.3f}")
```

Detalle de diseno: el subset compartido se filtra a `qa + repeat` porque `complete-en` y `complete-es` son tareas que solo existen en el dataset SFT-BPE — el char-level del cap 23 no podia generar palabras enteras razonables, asi que esas tareas no aparecian. Para que la comparacion sea honesta, los seis modelos se evaluan sobre los **mismos prompts** en las **mismas dos tareas**.

---

## 3. La tabla maestra

```
Shared eval examples: 500 (qa + repeat)

=== Tabla maestra: char-level vs BPE-level ===

modelo                qa        repeat    drift
----------------------------------------------------
char-base             0.000     0.000     0.500
char-sft              0.690     1.000     0.000
char-dpo              0.410     0.750     0.000
bpe-base              0.000     0.000     0.900
bpe-sft               0.190     0.770     0.000
bpe-dpo-b05           0.190     0.700     0.200
```

Tres columnas, seis filas, una verdad incomoda: en las metricas exact-match comparables, el **char-level gana**. La explicacion no es trivial, y desempaquetarla bien es lo que da el valor pedagogico del capitulo.

---

## 4. Lectura comparativa fila por fila

**`qa`: char-sft 69.0% vs bpe-sft 19.0%**. Char-level gana por 50 puntos porcentuales.

¿Por que? El BPE tokeniza `"Shakespeare\n"` como aproximadamente `['Shake', 'sp', 'eare', '\n']` — el modelo debe predecir esos cuatro tokens exactos en orden. Char-level predice once caracteres, secuencia mas larga pero **cada paso es entre 65 opciones de vocab**. La entropia por paso es:

$$
H_{\text{char}} = \log_2 65 \approx 4.2 \text{ bits/char}
$$
$$
H_{\text{bpe}} = \log_2 1112 \approx 10.1 \text{ bits/token}
$$

Aunque BPE compresa la respuesta a menos pasos, **cada paso es ~2.4× mas dificil de predecir** en bits crudos. Un dataset SFT pequeno (4000 ejemplos) y un modelo pequeno (Mini-LLaMA, 4 capas, 128 dim) no tienen capacidad para memorizar la asociacion `"Q: who wrote Hamlet?\nA: " → ['Shake','sp','eare','\n']` con la misma facilidad que `"Q: ...\nA: " → 'S','h','a','k','e','s','p','e','a','r','e','\n'`. Char-level memoriza por trayectoria; BPE exige discriminar entre miles de tokens a cada paso.

**`repeat`: char-sft 100.0% vs bpe-sft 77.0%**. Char-level otra vez gana, 23 pp.

El argumento es similar pero mas marcado. La tarea es `"INSTR: repeat 'a' three\nRESP: aaa\n"`. Char-level la resuelve casi trivialmente: ya emitio `'aaa\n'` mil veces en el dataset, son cuatro chars de un vocab de 65, la cadena es perfectamente predecible. BPE tokeniza `'aaa\n'` en algo como `['aaa', '\n']` o `['a', 'a', 'a', '\n']` segun como se haya construido el merge tree, y la estructura "repetir caracter N veces" no aprovecha la compresion de subwords — al reves, la rompe.

**`complete-en` y `complete-es`**: estan en el dataset BPE pero **no estan en el char-level**. Char-level no podia hacer estas tareas porque el modelo char-by-char con 4000 ejemplos no aprende fragmentos lexicos significativos. BPE las habilita arquitectonicamente, pero la accuracy exact-match es bajisima (0.5% en EN, 4–6.5% en ES) porque la tarea es semanticamente abierta y la metrica es brutalmente literal. Por eso no aparecen en la tabla compartida — solo aparecian en la tabla del cap 36, donde se mide con su propio benchmark.

**`drift`: char-base 50% vs bpe-base 90%**. El BPE-base es **mucho mas drifty**.

Lectura clave: `drift = 1 - tasa_de_seguir_el_prompt_instruccional`. Cuando le damos al modelo base un prompt como `"INSTR: capitalize 'cat'\nRESP: "`, el bpe-base genera continuaciones tipo "shall I lie thee, mine honest?" — palabras reales, gramatica reconocible, formato Shakespeare-perfecto. **Drift al 90%** significa que casi siempre genera estilo libro y casi nunca formato instruccional. El char-base hace algo distinto: produce caracteres que a veces se agrupan en palabras shakespeareanas, a veces en gibberish — la "coherencia shakespeareana" es mas porosa, asi que el detector de drift cuenta mas falsos negativos. **char-base 50%** no es mejor: es menos competente al estilo libro, lo que paradojicamente lo hace parecer menos drifty.

SFT lleva ambos a `drift = 0%` — el formato instruccional se aprende bien con el dataset SFT. DPO en char-level mantiene `drift = 0%`. DPO en BPE (`bpe-dpo-b05`) **sube drift a 20%** — un detalle importante. Es el unico setup donde DPO **empeora** el drift respecto al SFT. Hipotesis: el balance del beta sweep en cap 36 muestra que `beta=0.5` no fue suficiente para evitar que la policy empezara a re-aprender estilos del dataset DPO; los 1500 cross-task triples introducen "estilo de otra tarea como rejected", lo que puede mover marginalmente la policy hacia formatos que el evaluator de drift detecta. Es una regresion menor pero real.

---

## 5. La leccion central — cuando BPE, cuando char-level

Los numeros de arriba enganan si uno los mira sin pensar en para que existe cada tokenizacion. Char-level no existe para hacer "BPE pero peor"; existe para hacer **otras cosas**, mejor. Cuatro parrafos sobre cuando elegir cada uno.

**Char-level es perfecto para ensenar mecanismos.** Vocab pequeno (65 tokens en ingles, ~30 en idiomas con menos diacriticos), modelo que aprende rapido (loss baja en horas, no dias), y todo el setup se puede correr en un Mac sin GPU. Si uno quiere ver el efecto de loss masking, de RoPE, de la math de DPO — char-level hace que cada experimento tenga ciclo de feedback rapido. Eso es exactamente lo que hicimos en los caminos 1–2 de esta clase: char-level fue la herramienta pedagogica para construir cada pieza desde cero. **No es un sustituto de BPE en produccion** — es un microscopio que muestra el mecanismo que despues se aplica con BPE en serio.

**Char-level es artificialmente bueno en tareas tipo char-manipulation.** Reverse, repeat, capitalize, factoid memorizacion corta — todas son tareas donde el modelo no necesita semantica de palabras como unidad. Manipula caracteres directamente. Los 100% / 69% / 75% del char-level en `repeat`/`qa` no son evidencia de que "char-level es mejor" — son evidencia de que **el benchmark elegido es trivialmente facil para char-level**. Si el benchmark fuera "completa esta oracion en castellano natural", char-level fallaria sin remedio.

**BPE es necesario para tareas LLM reales.** Completacion, traduccion, conversacion, resumen, razonamiento sobre documentos — todo eso requiere que el modelo trate las palabras como unidad semantica. La palabra `"Madrid"` para un BPE es ~1 token con significado entero; para un char-level es 6 caracteres que el modelo debe agrupar internamente, gastando capacidad de la red en re-descubrir el concepto de palabra. `complete-en` y `complete-es` (que char-level no podia hacer) son la version minima de eso. Cualquier sistema serio de generacion de texto pasa por subwords.

**En produccion, BPE / SentencePiece es el estandar.** GPT-4 usa tiktoken (~100k tokens). Llama y Mistral usan SentencePiece (32k–128k segun version). Claude, Gemini, DeepSeek — todos subword. Char-level **sobrevive** en proyectos pedagogicos como este, en algunos modelos para idiomas con escritura no-segmentada (chino, tibetano antiguo) donde los segmentadores BPE tienen problemas, y en dominios super especificos (DNA, nombres de moleculas) donde los subwords no aportan mucho. Para todo lo demas — texto natural, asistentes conversacionales, modelos generativos de proposito general — la respuesta es BPE o derivados (tiktoken, SentencePiece, BBPE).

---

## 6. Cierre Camino 2.5 — recap completo

El Camino 2.5 cubrio el ciclo completo BPE de la clase:

- **[Cap 30]({{< relref "30-bpe-desde-cero" >}})**: BPE algoritmo desde cero, 1112 tokens de vocab bilingue (ingles + espanol).
- **[Cap 31]({{< relref "31-pretrain-bpe" >}})**: Pretrain Mini-LLaMA con BPE — loss baja de 7.18 a 2.68 sobre el corpus bilingue.
- **[Cap 32]({{< relref "32-refactor-tokenizer" >}})**: Refactor tokenizer-agnostic — `CharTokenizer` y `BPETokenizer` comparten interfaz, todo el resto del codigo no necesita saber cual es cual.
- **[Cap 33]({{< relref "33-dataset-sft-bpe" >}}) — [Cap 34]({{< relref "34-sft-bpe" >}})**: Dataset SFT-BPE (4 tareas, 4000 ejemplos) y SFT con loss masking. Resultados peores que char-SFT en metricas comparables, pero habilita `complete-en` / `complete-es`.
- **[Cap 35]({{< relref "35-dataset-dpo-bpe" >}}) — [Cap 36]({{< relref "36-dpo-bpe" >}})**: Dataset DPO-BPE con rejected linguisticamente ricos, beta sweep `{0.1, 0.5}`. Hipotesis cap 29 validada parcialmente — `beta=0.5` preserva mas accuracy.
- **Cap 37 (este)**: comparativa final, leccion sobre cuando usar cada tokenizacion.

Lo que sale del Camino 2.5 no es "construimos un mejor modelo" — es **"vimos exactamente que cambia cuando se cambia la tokenizacion, y cuando vale la pena pagar el costo"**.

---

## 7. Lo que sigue — caminos pendientes

Esta clase tiene varios caminos abiertos que se exploraran mas adelante en el curso:

- **Camino 3: Interpretabilidad mecanicista**. Tomar el Mini-LLaMA entrenado y abrirlo — analizar attention heads, identificar circuitos especializados, ver donde vive el conocimiento factual aprendido. La pregunta es "¿que aprendio realmente?".
- **Camino 4: BERT y modelos encoder-only**. MLM (masked language modeling), pre-training bidireccional, fine-tuning para clasificacion. Otra rama del arbol Transformer, optimizada para representacion en vez de generacion.
- **Camino 5: ViT (Vision Transformer)**. Aplicar la misma arquitectura a imagenes, patches en vez de tokens, primer cruce con el mundo multimodal.
- **Mini-experimentos**: usar `tiktoken` (50k+ vocab del GPT-4) sobre el mismo Mini-LLaMA para ver cuanto cambia con un vocab realista, datasets de preferencias humanas reales (UltraFeedback, HH-RLHF) para DPO con senal limpia, escalado a un Mini-LLaMA de 50M parametros.

Cada uno se construye encima del setup que cerramos aqui. El refactor del cap 32 permite enchufar cualquier tokenizador. Los modulos del cap 17–21 (RMSNorm, RoPE, GQA, KV-cache, SwiGLU) se reutilizan tal cual. La leccion del cap 22 (base sin SFT no sirve) y la del cap 24 (loss masking importa) viajan a cualquier camino siguiente.

---

## 8. Preguntas finales

1. **¿En que setting BPE-SFT seria estrictamente mejor que char-SFT?** Cuando la tarea pide respuestas largas, semanticamente ricas, en lenguaje natural fluido — completacion abierta, traduccion, resumen, conversacion. Char-level pierde porque (a) la longitud de generacion explota (cada palabra son 5–10 chars), (b) el modelo gasta capacidad re-aprendiendo el concepto de palabra desde cero, (c) la metrica exact-match deja de ser util y hay que pasar a perplexity / BLEU / juicio humano, donde el char-level pierde por falta de semantica. En cualquier dataset estilo MMLU, ARC, GSM8K, HumanEval — BPE gana sin discusion.

2. **¿Por que el bpe-base tiene drift 90% pero char-base solo 50%?** Porque drift mide "el modelo genera texto estilo libro en vez de seguir el prompt instruccional". Bpe-base genera oraciones completas, fluidas, gramaticalmente Shakespeare-perfectas — el detector de drift las identifica en casi todos los casos. Char-base genera caracteres que **a veces** se agrupan en palabras shakespeareanas y a veces en cadenas que ni siquiera son palabras reales — la "coherencia shakespeareana" es porosa, y el detector cuenta menos hits. **No es que char-base sea mejor instruccional**; es menos competente al estilo libro, lo que reduce su drift medido. Es un artefacto de la metrica, no un sintoma de virtud.

3. **¿Cuando preferirias char-level a BPE en un proyecto real?** Tres casos legitimos: (a) **proyectos pedagogicos para ensenar mecanismos** — exactamente este curso, donde el ciclo de feedback rapido importa mas que la accuracy maxima; (b) **modelos en idiomas con escritura no-segmentada o segmentacion ambigua** — chino antiguo, japones sin spaces, escrituras semiticas con vocalizacion variable, donde BPE puede crear merges semanticamente erroneos; (c) **dominios muy especificos donde subwords no aportan** — secuencias de DNA (4 simbolos), nombres SMILES de moleculas, codigo morse, sistemas formales con vocabulario super pequeno y bien definido. Fuera de esos tres casos, BPE/SentencePiece es la respuesta correcta por defecto, y char-level es una decision que necesita justificacion explicita.
