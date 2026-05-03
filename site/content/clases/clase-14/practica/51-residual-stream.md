---
title: "51 - Residual stream: la autopista del Transformer"
weight: 510
math: true
---

## 1. Apertura: el modelo no sobreescribe, suma

Hasta ahora pensabas el Transformer como una pila: bloque 1 transforma input, bloque 2 transforma output del bloque 1, etc. Pero esa abstraccion oculta la propiedad estructural mas importante de la arquitectura: cada bloque **no transforma** la representacion — la **lee y agrega un delta** sobre una autopista compartida.

Esa autopista es el **residual stream**. Tiene la dimension `d_model` y atraviesa el modelo de extremo a extremo. Cada bloque hace dos cosas:

1. **Lee** el stream actual (con atencion + FFN)
2. **Escribe** un delta al stream (con la conexion residual `x = x + bloque(x)`)

Ningun bloque tiene la autoridad de borrar lo que escribieron los anteriores. Todo lo que se escribio antes sigue ahi, posiblemente diluido pero presente. El head final lee la **suma acumulada** de todas las contribuciones.

Esta perspectiva — formalizada por Elhage et al. en "A Mathematical Framework for Transformer Circuits" (Anthropic 2021) — cambia como se piensa la interpretabilidad. Cada cabeza, cada FFN, cada bloque tiene un "canal de comunicacion" sobre el residual stream. Encontrar circuitos significa identificar quien escribe que en cada parte del stream y quien lo lee.

---

## 2. La conexion residual: ecuacion explicita

Cada `LLaMABlock` tiene la estructura:

$$x_{\text{post-attn}} = x + \text{Attn}(\text{RMSNorm}(x))$$
$$x_{\text{post-ffn}} = x_{\text{post-attn}} + \text{SwiGLU}(\text{RMSNorm}(x_{\text{post-attn}}))$$

El operador `+` es lo que crea la autopista. Ni `Attn` ni `SwiGLU` operan sobre el residual stream directamente — operan sobre una version normalizada (RMSNorm), y el resultado se SUMA al stream original.

Si quitaramos la conexion residual ($x = \text{Attn}(\text{RMSNorm}(x))$ sin el `+`), el stream se reescribiria en cada capa, perdiendo la informacion previa. Resnets (He et al. 2015) demostraron en vision que esa propiedad es lo que permite entrenar redes profundas; el Transformer hereda la idea.

Para el analisis de circuitos, la consecuencia es directa: **el stream en la capa N es la suma de las contribuciones de las capas 0 a N-1, mas las embeddings**. Mathematicamente:

$$x_N = \text{tok\_emb} + \sum_{i=0}^{N-1} \delta_i$$

donde $\delta_i$ es lo que el bloque $i$ escribio. Si pudieramos descomponer $\delta_i$ en sus componentes (cabeza por cabeza, FFN), sabriamos exactamente que aporto cada componente al output final. Eso es el programa de los siguientes capitulos.

---

## 3. Medicion empirica: ¿que tan grande es cada delta?

Para validar empiricamente que los bloques "agregan" al stream en lugar de "sobreescribirlo", comparamos:

- $||x||$: la norma del residual al ENTRAR al bloque
- $||y||$: la norma del residual al SALIR del bloque
- $||\delta|| = ||y - x||$: la norma de lo que el bloque escribio
- $||\delta|| / ||x||$: la magnitud relativa del delta vs lo que ya estaba
- $\cos(x, y)$: el coseno entre input y output (1.0 = misma direccion, 0 = perpendicular)

Si el bloque hace un edit suave, esperariamos `||delta|| << ||x||` y `cos ≈ 1`. Si sobreescribe, `||delta||` seria comparable o mayor a `||x||` y `cos` mucho menor.

---

## 4. Script

```python
"""51_residual_stream.py - Cap 51: residual stream como autopista del Transformer."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "To be or not to "
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

points = ["tok_emb"] + [f"blocks.{i}" for i in range(4)]
with cache_activations(model, points) as cache:
    with torch.no_grad():
        model(ids)

prev = cache["tok_emb"]
for i in range(4):
    cur = cache[f"blocks.{i}"]
    delta = cur - prev
    norm_in = prev.norm(dim=-1).mean().item()
    norm_out = cur.norm(dim=-1).mean().item()
    norm_delta = delta.norm(dim=-1).mean().item()
    rel = norm_delta / norm_in
    cos = torch.nn.functional.cosine_similarity(prev, cur, dim=-1).mean().item()
    print(f"block.{i}  ||in||={norm_in:.3f}  ||out||={norm_out:.3f}  "
          f"||delta||={norm_delta:.3f}  rel={rel:.3f}  cos={cos:.3f}")
    prev = cur
```

---

## 5. Output literal

```
=== El residual stream como autopista ===

Cada bloque LEE el stream actual y ESCRIBE una contribucion (delta).
La nueva activacion = activacion previa + delta del bloque.

Bloque       ||in||  ||out||  ||delta|| ||delta||/||in||  cosine(in,out)
----------------------------------------------------------------------
block.0       11.274    9.772      5.400            0.479           0.880
block.1        9.772   10.259      7.061            0.723           0.743
block.2       10.259   12.247      7.643            0.745           0.779
block.3       12.247   25.794     20.054            1.637           0.656

Interpretacion:
  - ||delta|| pequeno relativo a ||in|| -> el bloque modifica el stream con cuidado
  - ||delta|| grande -> el bloque sobreescribe partes del stream
  - cosine cercano a 1 -> el output preserva la direccion del input
  - cosine cercano a 0 -> el bloque rota el stream radicalmente

=== Diagrama ===

  tok_emb         block_0          block_1          block_2          block_3       norm_final -> head
     |               |                |                |                |
     v               v                v                v                v
  [emb] --------> [+d0] --------> [+d1] --------> [+d2] --------> [+d3] -> [norm]

cada bloque AGREGA un delta al stream sin sobreescribir.
los deltas se acumulan; el head final lee la suma de todas las contribuciones.
```

---

## 6. Analisis: lo esperado y lo sorprendente

### Lo esperado: bloques 0-2 hacen edits moderados

- `block.0`: delta/in = **0.48**, cosine = **0.88**. La primera capa edita el ~48% de la norma del stream pero mantiene un 88% de coseno con la entrada. Es un edit moderado: agrega informacion sin rotar la representacion.
- `block.1` y `block.2`: delta/in ≈ **0.72-0.75**, cosine ≈ **0.78**. Las capas medias escriben mas (deltas relativamente grandes) y rotan mas el stream. Esta es la fase donde el modelo construye representaciones ricas.

Hasta aqui, el stream se comporta como esperabamos: cada bloque AGREGA, no sobreescribe.

### Lo sorprendente: block.3 dispara la norma

El cuarto bloque tiene `||delta||/||in|| = 1.64` — **el delta es 1.64 veces mas grande que el residual de entrada**. La norma del stream pasa de 12.25 a 25.79 en una sola capa. Esto es muy distinto al comportamiento de los bloques anteriores.

¿Que significa? El cuarto bloque NO esta haciendo un edit suave — esta agregando una contribucion gigante que "domina" sobre todo lo que vino antes. Tres explicaciones posibles:

1. **El RMSNorm con `gamma` aprendido permite amplificar**: los parametros gamma de la RMSNorm pueden ser arbitrariamente grandes, y la capa puede aprender a usarlos para producir activaciones de gran magnitud. El cuarto bloque parece haber aprendido eso.

2. **El bloque escribe directamente al output**: no hay nada despues del bloque 3 excepto la norm final + head. Cualquier informacion critica para la prediccion del siguiente token tiene que llegar al stream en este punto. El modelo aprendio a usar el bloque final como "gran escritor" antes de la salida.

3. **La norma final compensa**: justo despues del bloque 3 viene `norm_final` (RMSNorm), que normaliza el stream para que el head pueda usarlo. Sin esa norm, los logits de la cabeza serian numericamente inestables. RMSNorm le permite al modelo "escribir grande" en el bloque 3 sabiendo que despues sera renormalizado.

El coseno tambien cae a **0.66** en el bloque 3 — el output preserva el 66% de la direccion del input pero introduce mucha rotacion. Este bloque cambia la representacion mas radicalmente que los anteriores.

### La conclusion honesta

El residual stream NO es uniforme: los primeros bloques hacen edits suaves, los del medio hacen edits moderados, y el ultimo dispara la magnitud. Esta heterogeneidad es **importante** para la interpretabilidad: cuando busquemos circuitos en proximos capitulos, las cabezas del ultimo bloque tendran efectos desproporcionadamente grandes sobre el output. No por arquitectura — por como el modelo aprendio a usar la arquitectura.

---

## 7. La autopista en perspectiva: el residual stream como "memoria de trabajo"

Una manera util de pensar el residual stream es como la **memoria de trabajo** del modelo. En cada posicion, el vector de `d_model=128` dimensiones representa "todo lo que el modelo sabe" sobre ese token, integrado a lo largo de la secuencia y a lo largo de las capas.

- Los primeros bloques escriben **features locales** (que caracter es, que tipo de palabra esta empezando).
- Las capas medias escriben **features de contexto** (es un sustantivo en posicion de sujeto, esta dentro de una linea de Brutus, etc.).
- El ultimo bloque escribe **features de prediccion** (cual es probable que sea el siguiente token).

Esta jerarquia no es un postulado teorico — es lo que veremos en los proximos capitulos cuando apliquemos:

- **Cap 52 (logit lens)**: proyectar el residual intermedio al vocab para ver que predice cada capa
- **Cap 56 (QK/OV)**: descomponer las cabezas en lo que LEEN y lo que ESCRIBEN
- **Cap 60 (SAE)**: descomponer el stream en features interpretables individuales

Todos estos analisis tienen sentido porque el residual stream es UN solo vector, accesible a todos los componentes, donde toda la informacion vive y se acumula.

---

## 8. Preguntas de verificacion

**1. ¿Por que la conexion residual es necesaria para entrenar redes profundas?**

Sin la conexion residual, los gradientes durante backprop tendrian que atravesar todas las transformaciones del bloque (atencion + FFN + normas) para llegar a los pesos de la capa anterior. En cada paso, la magnitud del gradiente puede crecer o decrecer multiplicativamente; en redes profundas (10+ capas), esto causa **vanishing/exploding gradients**: el gradiente que llega a la primera capa es practicamente cero o astronomico, y el entrenamiento falla. La conexion residual crea un **camino directo** para el gradiente: $\partial x_N / \partial x_0$ siempre tiene un termino $1$ que pasa por el "atajo" de los `+`. Esto garantiza que los gradientes lleguen a todas las capas con magnitud controlada. ResNet (He et al. 2015) demostro este principio en vision; el Transformer lo hereda.

**2. ¿Que pasaria si el bloque 3 tuviera `gamma` muy chico en su RMSNorm?**

RMSNorm escala el stream por un parametro aprendido `gamma` (el "gain"). Si gamma fuese muy chico (digamos 0.01) en el bloque 3, las contribuciones del bloque (atencion + FFN, que reciben input normalizado por RMSNorm) serian tambien muy chicas — porque el input a estas operaciones se multiplica por gamma. El delta del bloque 3 seria tiny, y el stream llegaria al head con norma ~12 (similar a block.2). El head, que aprendio a operar con normas mayores (post-norm-final que reescala), recibiria activaciones desproporcionadamente pequenas y produciria predicciones erroneas. La leccion: los `gamma` de las normas son una palanca que el modelo usa para regular la "amplitud de escritura" de cada bloque.

**3. Si el bloque 3 escribe tan agresivamente, ¿por que no se "borran" las contribuciones de los bloques anteriores?**

Aqui esta la sutileza: el bloque 3 escribe una contribucion grande, pero **AGREGA** al stream — no lo reemplaza. Despues del bloque 3, el stream contiene `tok_emb + delta_0 + delta_1 + delta_2 + delta_3` (suma vectorial). Que `||delta_3||` sea grande significa que esa contribucion domina **en norma**, pero los componentes anteriores siguen presentes. El head final puede leer subespacios donde los deltas anteriores son significativos, mientras que otras componentes del stream (donde `delta_3` escribio) capturan informacion fresca. La interpretabilidad mecanicista (caps 56, 60) descompone el stream en estos subespacios para identificar quien escribio que.
