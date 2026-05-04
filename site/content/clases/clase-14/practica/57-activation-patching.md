---
title: "57 - Activation patching: del correlacional al causal"
weight: 570
math: true
---

## 1. Apertura: causalidad, no correlacion

Los caps 53-56 fueron descriptivos: que cabezas tienen patrones, donde miran, como se descomponen matematicamente. Pero ninguno responde la pregunta clave: **¿cuales componentes son CAUSALMENTE necesarios para la prediccion del modelo?**

Una cabeza puede tener un patron clarisimo (previous-token con score 0.55) y aun asi ser irrelevante para tareas concretas. Una FFN puede tener pesos enormes y no contribuir nada al output final. Sin un test causal, los patrones que vemos son hipotesis no validadas.

**Activation patching** (Geiger et al. 2021, Wang et al. 2022) es la tecnica canonica para tests causales en redes neuronales. La idea es brutal en su simplicidad:

1. Correr el modelo sobre un prompt **clean**, cachear todas las activaciones
2. Correr sobre un prompt **corrupted** que da una prediccion distinta
3. Para cada componente (capa, posicion), reemplazar la activacion del corrupted con la del clean y medir cuanto cambia la prediccion

Si reemplazar el componente X **restaura** la prediccion clean, X es causalmente importante. Si no cambia nada, X es prescindible para esa decision.

Esta es la tecnica que descubrio el circuito IOI en GPT-2 (Wang et al. 2022): 26 cabezas trabajando juntas para resolver "John gave Mary a flower. Mary gave a flower to ___". Vamos a aplicarla a Mini-LLaMA con un experimento mas modesto.

---

## 2. El experimento: speaker identity

Diseñamos un par clean/corrupted donde solo cambia el speaker:

- **Clean**: `"BRUTUS:\nI am "`
- **Corrupted**: `"BIANCA:\nI am "`

Mismo numero de tokens (13), misma estructura. Lo unico que cambia son las posiciones 1-5: `RUTUS` vs `IANCA`. Si el modelo "sabe" que speaker es Brutus vs Bianca y eso afecta su prediccion del siguiente caracter, lo veremos como diferencia entre los logits clean y corrupted.

### El target

Sobre estos prompts, el modelo predice el mismo top-1 en ambos casos (`'a'`, presumiblemente para empezar palabras como "afraid" o "angry"). Pero los **logits** difieren: el logit del caracter `'n'` (que podria empezar "noble", "not", etc.) es +4.01 en clean y +3.58 en corrupted. Una diferencia de 0.43 en logit significa que el modelo trata "BRUTUS" y "BIANCA" como contextos distintos.

Usamos `'n'` como target: medimos cuanto recovery del diff = 0.43 logra cada patch.

---

## 3. Metrica: recovery score

Para cada celda `(layer, position)`:

$$\text{recovery} = \frac{\text{patched\_logit} - \text{corrupted\_logit}}{\text{clean\_logit} - \text{corrupted\_logit}} \times 100\%$$

- **0%**: el patch no cambio nada — el componente NO es causal
- **100%**: el patch restauro completamente la prediccion clean
- **>50%**: el componente es altamente causal
- **negativo**: el patch empeoro las cosas — el componente "lucha" contra la prediccion clean

Patcheamos el OUTPUT de cada bloque en cada posicion, una a la vez. Esto da un grid `4 capas × 13 posiciones = 52 celdas`.

---

## 4. Script

```python
"""57_activation_patching.py - Cap 57: activation patching para causalidad."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, patch_activation

torch.manual_seed(1337)
device = get_device()
tok = CharTokenizer(load_text("shakespeare.txt"))
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

clean_ids = torch.tensor([tok.encode("BRUTUS:\nI am ")], dtype=torch.long, device=device)
corrupted_ids = torch.tensor([tok.encode("BIANCA:\nI am ")], dtype=torch.long, device=device)

# Cachear activaciones del clean
with cache_activations(model, [f"blocks.{i}" for i in range(4)]) as clean_cache:
    with torch.no_grad():
        clean_logits, _ = model(clean_ids)

with torch.no_grad():
    corrupted_logits, _ = model(corrupted_ids)

target_id = (clean_logits[0, -1] - corrupted_logits[0, -1]).argmax().item()
diff = (clean_logits[0, -1, target_id] - corrupted_logits[0, -1, target_id]).item()

def patching_score(layer, pos):
    name = f"blocks.{layer}"
    clean_act = clean_cache[name][:, pos:pos+1, :]
    patched = patch_activation(model, corrupted_ids, {name: (pos, clean_act)})
    return ((patched[0, -1, target_id].item() - corrupted_logits[0, -1, target_id].item()) / diff) * 100

# Grid de patching
for layer in range(4):
    for pos in range(13):
        score = patching_score(layer, pos)
        print(f"block.{layer} pos.{pos}: {score:+.1f}%")
```

---

## 5. Output literal

```
Clean:     'BRUTUS:\nI am '
Corrupted: 'BIANCA:\nI am '
T = 13 tokens

Clean prediction:     'a' (id=39)
Corrupted prediction: 'a' (id=39)

ADVERTENCIA: clean y corrupted predicen lo mismo. Patching no informativo.
Buscando la mayor diferencia en logits...
Token con mayor diff (clean - corrupted): 'n'

Logit del target 'n' en:
  clean:     +4.007
  corrupted: +3.582
  diff (clean - corrupted) = +0.425

=== Activation patching: % de recovery por (layer, posicion) ===

             B     R     U     T     U     S     :    \n     I           a     m      
block.0      +0    +5    +9   -18   +18    -9    +1    -2    +1    +0   +15   +20   +40
block.1      +0    +9    -1   -12    +0    -5    +4    -1   +13    -1   +29    +9   +45
block.2      +0    +2    +0    +2    -6    -3    +7    -1    -2    -0    +3   +15   +80
block.3      +0    +0    +0    +0    +0    +0    +0    +0    +0    +0    +0    +0  +100

Cells con recovery > 30%:
  block.0 pos.12 (token=' ')  recovery=+40.5%
  block.1 pos.12 (token=' ')  recovery=+45.3%
  block.2 pos.12 (token=' ')  recovery=+80.2%
  block.3 pos.12 (token=' ')  recovery=+100.0%
```

---

## 6. Analisis: el flujo causal va por la posicion final

### El patron dominante: columna 12

La unica columna con recovery alto y monotono es la **posicion 12** — el trailing space, donde el modelo hace su prediccion. Recovery sube de 40% (capa 0) a 100% (capa 3). Esto es esperable: el residual stream final en la posicion 12 contiene la prediccion final, asi que patchearlo COMPLETAMENTE restaura la salida clean.

Lo informativo es la **velocidad de recovery**: del 40% en capa 0 ya el residual de la posicion 12 lleva el 40% de la informacion necesaria. Esto significa que las primeras dos capas ya "transmitieron" mucha info del speaker hacia la posicion final via atencion.

### Posiciones 0-5: el speaker se procesa pero su efecto es pequeno

Las posiciones 1-5 (donde clean y corrupted difieren) muestran efectos modestos: recovery entre -18% y +18% en bloques 0-1, y casi 0% en bloques 2-3. Lectura:

- En **capa 0**, patchear el speaker tiene efectos del orden de ±10-20%. La capa esta procesando la identidad del speaker pero su efecto causal directo sobre la prediccion es chico.
- En **capas 2-3**, los efectos en posiciones del speaker desaparecen (~0%). El procesamiento ya no esta "alli" — todo se movio a la posicion 12 via la atencion.

**Esto es flujo causal en accion**: la informacion del speaker entra al modelo en posiciones 1-5 (capa 0), se transporta via atencion a la posicion 12 (capa 1), y se cristaliza ahi (capas 2-3).

### El sign mixto en capa 0

`block.0` tiene celdas tanto positivas (+9, +18) como negativas (-18, -9, -12). Algunas posiciones del speaker AYUDAN a recuperar clean cuando se patchean; otras EMPEORAN la situacion.

¿Por que? La capa 0 mezcla informacion en formas que pueden ser anti-correlacionadas con el target final. Patchear una posicion no afecta solo "esa posicion" — la atencion en la posicion 12 mira hacia atras, mezclando todas las posiciones procesadas. Patchear posicion 4 con la activacion clean reemplaza una pieza del puzzle que la atencion en 12 lee, pero esa pieza puede no encajar bien con las otras piezas no-patcheadas.

Esto es por que **circuit discovery** (cap 58) requiere patchear conjuntos de componentes simultaneamente, no uno a uno.

### Pos 11 = 'a' tiene recovery moderado

`block.0 pos.11 = +20%` y `block.1 pos.11 = +9%`. La 'a' del prompt (posicion 11) tiene efecto causal — patchearla con su version clean recupera 20% del logit. Esto sugiere que el residual en esa posicion lleva info que la atencion en 12 lee.

### El recovery exacto de 100% en block.3 pos.12 es trivial

Es esperado matematicamente: patchear el residual stream final justo antes del head equivale a sustituir el output del modelo. No es un descubrimiento, es validacion del pipeline.

---

## 7. Lo que esto valida y lo que NO valida

### Valida

- **El pipeline de patching funciona**: el recovery 100% en el caso trivial confirma que `patch_activation` esta correctamente implementado.
- **La info del speaker fluye principalmente a la posicion 12**: esto es un descubrimiento causal real, no correlacional.
- **Las primeras capas hacen el trabajo de transporte**: en capa 0 ya hay 40% del recovery en pos 12; en capa 2 ya hay 80%.

### No valida

- **Cuales cabezas especificas son responsables**: este patching opera a nivel de bloque (output completo del block). Para identificar cabezas individuales necesitariamos patcheo a nivel de cabeza (cap 58 lo hace).
- **Que features del residual stream son las relevantes**: el bloque escribe `(d_model = 128)` valores; cuales de esas dimensiones llevan la info del speaker es invisible a este nivel.
- **Si hay un circuito discreto**: patcheo posicion-por-posicion es coarse. El circuito IOI requiere patcheo de conjuntos.

---

## 8. La diferencia con cap 56 (descomposicion QK/OV)

Cap 56 fue **estructural y no-causal**: extrajo matrices `W_Q W_K^T` de la cabeza top y analizo sus singular values, su efecto sobre embeddings, etc. Concluyo que la cabeza NO es copy head. Pero NO probo que la cabeza sea importante para alguna tarea.

Cap 57 es **causal y agregado**: no descompone, intervene. Mide el efecto del bloque completo sobre la prediccion final. Concluye que el speaker procesado en posiciones 1-5 fluye hacia la posicion 12 via las primeras dos capas.

Las dos tecnicas son complementarias:

- Decomposition responde "**que computa cada cabeza?**"
- Patching responde "**que componentes importan para la tarea?**"

Combinarlas — descomponer cabezas que el patching identifico como causales — es el siguiente paso (cap 58).

---

## 9. Preguntas de verificacion

**1. ¿Por que clean y corrupted predicen el mismo top-1 ('a')?**

Mini-LLaMA char-level tiene 4 capas y ~890K parametros. La identidad del speaker (Brutus vs Bianca) es una distincion semantica de alto nivel — requiere el modelo "comprender" que cada speaker tiene patrones de habla distintos. A esta escala, el modelo aprendio estadisticas char-level dominantes (despues de espacio en linea de dialogo viene una vocal, frecuentemente 'a'), pero no las distinciones especificas por speaker. Por eso el top-1 es el mismo para ambos prompts. La diferencia ESTA — vista en logits — pero es chica (0.43 en logit) y no flippa el ranking. En modelos mas grandes (GPT-2 small+) sobre datos mas variados, la identidad del speaker SI cambiaria predicciones top-1.

**2. ¿Por que algunas celdas tienen recovery negativo?**

Recovery negativo significa: patchear ese componente con la version clean **empeora** la prediccion clean en el corrupted run. Esto ocurre cuando el componente patcheado interactua con OTROS componentes no-patcheados. Imaginate dos piezas A y B que en clean trabajan juntas (A ayuda + B ayuda = +1.0 al logit del target). Si patcheas solo A en el corrupted, A_clean trabaja con B_corrupted que no espera a A_clean — el resultado puede ser peor que el corrupted original. La leccion: el patching uno-a-uno detecta efectos LINEALES; los efectos no-lineales (interacciones entre componentes) requieren patcheo simultaneo.

**3. ¿Que significaria un grid de patching donde casi todas las celdas son ~0% excepto la columna final?**

Significaria que el modelo NO procesa nada de las posiciones intermedias para la tarea — solo la posicion final importa. Eso seria sintomatico de un modelo "lazy" que ignora el contexto y predice basandose solo en el ultimo token. Mini-LLaMA no es asi: vimos efectos significativos en pos 11 (la 'a') y patrones mixtos en pos 1-5 (el speaker). Esto confirma que el modelo SI integra informacion de posiciones intermedias, aunque la integracion final ocurre en la posicion 12. El grid de patching es una herramienta diagnostica para detectar cuanto contexto realmente usa el modelo.
