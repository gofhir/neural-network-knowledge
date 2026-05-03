---
title: "52 - Logit lens: predicciones capa por capa"
weight: 520
math: true
---

## 1. Apertura: ¿que predice el modelo a media procesamiento?

Cap 50 mostro como cachear activaciones intermedias. Cap 51 mostro que el residual stream acumula contribuciones a lo largo de las capas. La pregunta natural: **¿que ve el head si lo aplicamos a un residual intermedio en lugar del final?**

La tecnica se llama **logit lens** (nostalgebraist 2020). La idea es asombrosamente simple:

```
logits_intermedios = head(norm_final(residual_de_capa_k))
```

Aplicas el head del modelo al residual stream **despues de la capa k**, en lugar de despues de todas las capas. Cada capa tiene asi su "prediccion provisional" del siguiente token. Comparando estas predicciones a lo largo del modelo, podemos rastrear como emerge la respuesta capa por capa.

El objetivo: ver el modelo "pensar". ¿Cuando aparece la respuesta correcta? ¿Cambia drasticamente entre capas o evoluciona suave? ¿Una capa intermedia "casi acierta" antes de que las siguientes la alteren?

---

## 2. Por que funciona: el residual stream tiene la dimension correcta

El head del modelo es una sola matriz `(d_model, vocab_size)` que proyecta el residual stream al vocabulario. La capa final del modelo aplica `head(norm_final(stream))` para producir logits.

Pero la matriz del head no "sabe" si el stream que le llega viene de la capa 1, la capa 3, o la final — solo opera sobre vectores de `d_model` dimensiones. Si le pasas un residual intermedio, te da logits validos sobre el vocab. Esos logits no necesariamente representan la prediccion final del modelo, pero representan **lo que el modelo predeciria si la capa actual fuera la ultima**.

La unica sutileza es que el modelo aplica `norm_final` antes del head. Para que la comparacion sea justa, debemos normalizar tambien los residuales intermedios con la misma `norm_final`. Sin eso, los residuales intermedios (con norma distinta a la del residual final) producirian logits con magnitudes erroneas.

```python
def logit_lens(model, residual):
    """Aplica head al residual stream para proyectar al vocab.
    El residual debe ser POST-norm final."""
    return model.head(residual)

# Uso:
h_intermedio = cache["blocks.2"]
h_normalizado = model.norm_final(h_intermedio)
logits_capa_2 = logit_lens(model, h_normalizado)
```

---

## 3. La prediccion correcta de Shakespeare: una hipotesis

Sobre el prompt `"To be or not to "`, la continuacion humana obvia es "be" — la frase celebre de Hamlet. El modelo char-level fue entrenado sobre el corpus completo de Shakespeare, asi que en principio "ha visto" esa frase miles de veces.

Hipotesis: las primeras capas predeciran caracteres genericos comunes (vocales, espacios), las capas intermedias empezaran a converger hacia `'b'` cuando integren el contexto suficiente, y las ultimas refinaran la prediccion. La caracteristica observada en cap 51 — que el bloque 3 escribe agresivamente al stream — sugiere que es ahi donde la decision final se cristaliza.

Veamos que pasa realmente.

---

## 4. Script

```python
"""52_logit_lens.py - Cap 52: predicciones capa por capa via logit lens."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, logit_lens

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "To be or not to "
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

points = ["tok_emb"] + [f"blocks.{i}" for i in range(4)] + ["norm_final"]
with cache_activations(model, points) as cache:
    with torch.no_grad():
        model(ids)

target_id = tok.encode("b")[0]
for name in points:
    h = cache[name]
    h_norm = model.norm_final(h)
    logits = logit_lens(model, h_norm)
    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    top = probs.topk(3)
    p_target = probs[target_id].item()
    rank = (probs > probs[target_id]).sum().item() + 1
    print(f"{name}  top-1={tok.decode([last.argmax()])!r}  P(b)={p_target:.4f}  rank(b)={rank}")
```

---

## 5. Output literal

```
Prompt: 'To be or not to '
Posicion final del stream: 15 (proxima prediccion)

=== Top-3 predicciones para la posicion final, capa por capa ===

Punto         top-1                  top-2                  top-3                 
--------------------------------------------------------------------------------
tok_emb       'r'=0.118              'a'=0.081              'b'=0.054             
blocks.0      'a'=0.153              'r'=0.147              's'=0.125             
blocks.1      's'=0.126              'm'=0.116              'a'=0.090             
blocks.2      't'=0.287              's'=0.132              'b'=0.095             
blocks.3      't'=0.117              's'=0.097              'b'=0.089             
norm_final    't'=0.113              's'=0.093              'b'=0.089             

=== Evolucion de la prediccion top-1 vs el target probable 'b' (de 'be') ===

target='b' id=40
Punto             P(b)    rank de b top-1 actual   
--------------------------------------------------
tok_emb         0.0543            3 'r'            
blocks.0        0.0620            6 'a'            
blocks.1        0.0789            4 's'            
blocks.2        0.0954            3 't'            
blocks.3        0.0894            3 't'            
norm_final      0.0894            3 't'
```

---

## 6. Analisis honesto: el modelo NO predice 'be'

Esto es un resultado pedagogicamente importante: **el modelo char-level entrenado en Shakespeare NO predice `'b'` como el siguiente caracter despues de `"To be or not to "`**. Predice `'t'` con probabilidad ~11%.

¿Por que? El modelo ve el prompt como una secuencia de 16 caracteres que termina en espacio. La prediccion mas probable despues de un espacio en Shakespeare no es "b" — es "t". La t es una de las consonantes mas frecuentes para empezar palabras en ingles ("the", "to", "that", "thou", "thy", "thee"). El modelo aprendio estadisticas char-level: dado un espacio, predice la consonante inicial mas comun.

La frase "To be or not to be" es una idiom famosa para humanos, pero para un modelo char-level con 4 capas y ~890K parametros, integrar el contexto distante (que ya vimos "To be" hace 14 caracteres atras) y predecir "be" otra vez requiere capacidad de razonamiento que esta arquitectura simplemente no tiene a esta escala.

### Lo que el logit lens si revela: la evolucion capa por capa

Aunque la prediccion final no es `'b'`, la evolucion de las predicciones intermedias es informativa:

| Punto | Top-1 | Probabilidad |
|---|---|---|
| `tok_emb` | `'r'` | 11.8% |
| `blocks.0` | `'a'` | 15.3% |
| `blocks.1` | `'s'` | 12.6% |
| `blocks.2` | `'t'` | **28.7%** |
| `blocks.3` | `'t'` | 11.7% |
| `norm_final` | `'t'` | 11.3% |

- **`tok_emb`**: la prediccion antes de cualquier procesamiento es `'r'` con 11.8%. Esto es el "embedding lens": que predice si solo usaramos las embeddings sin atencion ni FFN. La respuesta refleja correlaciones brutas entre el caracter actual (espacio) y los siguientes caracteres en el corpus.
- **`blocks.0` y `blocks.1`**: el top-1 cambia entre capas (`'a'` y `'s'`). Las primeras capas estan reorganizando las predicciones pero sin converger.
- **`blocks.2`**: salto cualitativo — el top-1 es `'t'` con **28.7%**. Esta capa "decide" que viene una palabra que empieza con t. Es la capa con la prediccion mas concentrada de todas.
- **`blocks.3` y `norm_final`**: la confianza en `'t'` cae a ~11%, pero `'t'` sigue siendo top-1. El bloque final SUAVIZA la prediccion del bloque 2 — agrega ruido o alternativas. Esto es coherente con lo visto en cap 51: el bloque 3 escribe agresivamente al stream, posiblemente agregando informacion que dispersa la masa de probabilidad.

### El rank de 'b' no llega al podio

Si miramos especificamente que rank tiene `'b'` capa a capa: empieza en rank 3, baja a rank 6 en blocks.0 (peor!), y termina en rank 3. Nunca llega al top-1 ni al top-2. La probabilidad de `'b'` se mantiene entre 5% y 10% a lo largo del modelo — sube modestamente pero nunca se convierte en la prediccion mas probable.

### La leccion pedagogica

Logit lens es una tecnica poderosa **cuando el modelo realmente sabe predecir bien**. Sobre Mini-LLaMA char-level entrenado pocos pasos sobre Shakespeare, las predicciones son borrosas en todas las capas. La tecnica funciona — captura predicciones validas — pero el modelo no tiene la capacidad para producir predicciones nitidas.

En modelos grandes (GPT-2, GPT-3, etc.) y prompts donde el siguiente token es altamente predecible, el logit lens muestra patrones mas claros: el target aparece como top-1 en alguna capa intermedia y se queda asi hasta el final, o las capas tempranas predicen cosas relacionadas (sinonimos, mismo tipo de palabra) que se refinan en capas posteriores. Aqui no vemos eso — vemos un modelo pequeno que apenas distingue entre los caracteres mas frecuentes.

Esta honestidad es importante: la interpretabilidad mecanicista funciona mejor sobre modelos competentes. Mini-LLaMA char-level es nuestro laboratorio pero tiene limitaciones de escala que afectan que tan limpios son los patrones que vemos.

---

## 7. Logit lens en la practica industrial

En modelos grandes, logit lens se usa para:

- **Diagnosticar capas problematicas**: si el rank del target sube monotonamente capa a capa hasta una capa donde de pronto baja, sospechamos que esa capa "rompe" la prediccion. Util para debug de fine-tuning donde el comportamiento se degrada.
- **Identificar el "salto" de procesamiento**: en GPT-2 small, ciertas tareas (predecir el genero de un nombre, completar capitales) saltan al top-1 en una capa especifica. Esa capa es donde reside el circuito.
- **Validar circuit hypotheses**: si descubres un circuito hipotetico (caps 56-58), logit lens te permite verificar que las capas implicadas efectivamente cambian la prediccion en la direccion esperada.
- **Tuned lens**: una variante (Belrose et al. 2023) entrena una transformacion lineal especifica para cada capa que mejora la precision del lens. Util cuando la `norm_final` no es suficiente.

Para Mini-LLaMA, logit lens es mas un "termometro" que un "telescopio": confirma que el modelo procesa secuencialmente pero no revela circuitos por su limitacion de escala.

---

## 8. Preguntas de verificacion

**1. ¿Por que aplicamos `model.norm_final` al residual intermedio antes del head?**

La capa final del modelo aplica `head(norm_final(stream))`. Si pasaramos el residual intermedio directamente al head sin normalizar, los logits tendrian magnitudes muy distintas (porque la norma del stream cambia entre capas — vimos en cap 51 que va de 9 a 26). La softmax, al ser sensible a la magnitud, daria distribuciones artificialmente picudas o achatadas que no reflejan lo que el modelo "predice" en esa capa. Aplicar `norm_final` normaliza el residual a la misma escala que el residual final, haciendo la comparacion justa. Esta es la convencion estandar de logit lens (nostalgebraist 2020).

**2. Si el modelo no predice 'b' despues de "To be or not to ", ¿significa que el logit lens es defectuoso?**

No. El logit lens captura honestamente lo que el modelo predice — y el modelo simplemente no es lo suficientemente bueno para predecir esa frase iconica. La tecnica funciona; el modelo es chico. Si entrenaramos un modelo mas grande (12 capas, d_model=512) sobre el mismo corpus durante mas pasos, probablemente veriamos el rank de 'b' subir al top-3 o top-1 en alguna capa intermedia. La leccion es que la utilidad del logit lens depende de la competencia del modelo en la tarea evaluada — si el modelo no puede resolver la tarea, el lens lo confirma sin embellecerlo.

**3. ¿Que pasaria si el head del modelo no fuera lineal sino una red neural pequena?**

El head linear (`nn.Linear(d_model, vocab)`) es lo que hace que el logit lens sea matematicamente directo: aplicar la misma matriz al residual intermedio te da logits validos. Si el head fuera no-lineal (ej: MLP), aplicarlo a un residual intermedio NO seria lo que el modelo "predice en esa capa" — seria lo que un MLP que opera sobre activaciones POST-norm-final predice cuando le das activaciones PRE-procesadas. La distincion es importante: el head lineal es una funcion del residual stream final que tambien es valida sobre residuales intermedios (porque ambos viven en el mismo espacio `d_model`). Un head no-lineal romperia esa propiedad. Esta es una de las razones por las que la mayoria de Transformers usan heads lineales — facilita la interpretabilidad ademas de tener menos parametros.
