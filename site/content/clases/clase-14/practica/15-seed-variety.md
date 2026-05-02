---
title: "15 - Variedad con seeds: por que cada conversacion es unica"
weight: 150
math: true
---

Llegamos al ultimo experimento de la serie. Despues de explorar profundidad, ancho, contexto, temperatura y top-k, queda una pregunta que toca la fibra mas filosofica de los LLMs: **¿por que cada vez que le hago la misma pregunta a ChatGPT me responde algo distinto?** No es un bug. No es que el modelo "cambie de opinion". Es algo mucho mas estructural: la **semilla aleatoria** del sampling. Este capitulo lo desarma.

El script que acompana este capitulo es `clase_14/practica/12_seed_variety.py`. Es el ultimo del viaje.

---

## 1. El experimento

Mismo modelo. Mismo prompt. Misma temperatura. Misma top-k.

¿Que cambia? La **semilla aleatoria** (`torch.manual_seed`).

Todo lo demas se mantiene fijo: los pesos del modelo entrenado, el prompt `ROMEO:`, `temperature=0.8`, `top_k=10`. La unica perilla que tocamos es la seed antes de cada generacion. El resultado es 5 textos completamente distintos, todos plausiblemente Shakespearianos.

```python
for seed in [0, 42, 100, 1234, 9999]:
    torch.manual_seed(seed)
    print(sample(model, tokenizer, "ROMEO:\n",
                 max_new_tokens=200, temperature=0.8, top_k=10))
```

Cinco lineas de codigo. Cinco universos paralelos.

---

## 2. La salida real

5 generaciones del prompt `ROMEO:`, cada una con una seed distinta:

### seed = 0

```
ROMEO:
I too have an them shreek have hamber'd
Whoping with thee of the sout of thee,
Whut, sook one' to thus be oly shing burd a cousin
To me brook our hither's mest of speech the cause to--
First my here:
```

### seed = 42

```
ROMEO:
Troy, make of the partion's common that to that the
poin'd me thy head's thou art were of his these.
Had more he belood?

LEONTES:
O, we thee, I ceasing more,
And tall to the canstel as the pling, and
```

### seed = 100

```
ROMEO:
Why, be say him well were he's a sun decon in
That she out such out to choild beling and out
That man are courtion thou couldsinst by heaves of me is count
To the prince so contruppent to tembroans. B
```

### seed = 1234

```
ROMEO:
Will which the had offire of thy wretch,
I speak all marition's but his mind broved me
to cruss mock and which streed, by all the hown of
Alange one as that me, here hears are with a puison;
And thou
```

### seed = 9999

```
ROMEO:
If mine a wordshops; I am my geant,
And sir, there of you deside were oven hold of the
Affide a benour sayst our hast would herse thou and to the
To seemir; but my torchall doth the peain.

FROMEY:
I
```

Las 5 son del mismo modelo, con los mismos hyperparametros de sampling. La unica diferencia: el numero entero que pasamos a `torch.manual_seed`.

---

## 3. Que esta pasando

El modelo es **completamente deterministico** dada una seed. Lo no-deterministico es el **muestreo del softmax**. La seed controla esa aleatoriedad.

```python
# La operacion no-deterministica:
idx_next = torch.multinomial(probs, num_samples=1)
```

`multinomial` toma una distribucion (vector de probabilidades que suma 1) y devuelve un sample — un indice elegido con probabilidad proporcional a su peso. La seed inicializa el RNG (random number generator) que decide cual.

{{< concept-alert type="clave" >}}
**Misma distribucion + distinta seed = sample distinto.** Esa es la base de toda la variedad de los LLMs. El modelo no "piensa diferente" entre llamadas — siempre produce la misma distribucion para el mismo contexto. Lo que cambia es que dado de las probabilidades sale en cada tirada.
{{< /concept-alert >}}

Una vez que el primer token sale distinto, el contexto cambia, la distribucion para el segundo token cambia, y a partir de ahi las dos generaciones divergen totalmente. Es un sistema **caotico**: pequenas perturbaciones iniciales (un token distinto) producen trayectorias completamente distintas.

---

## 4. Por que esto importa: la imprevisibilidad de los LLMs

Cuando le haces la misma pregunta dos veces a ChatGPT, **NO** vas a recibir la misma respuesta. ¿Por que?

1. La API tiene una seed aleatoria que cambia cada llamada (tipicamente derivada del clock del sistema).
2. El softmax sobre el vocab tiene muchas probabilidades no-cero — varios tokens son "plausibles" en cada paso.
3. Distinta seed → distinto sample → distinta cadena de tokens → distinta respuesta.

Es **estructural**, no un bug. El modelo es estocastico por diseno.

Esto tiene consecuencias practicas grandes:

- **Reproducibilidad cientifica**: si publicas un paper donde "GPT-4 hace X", otro investigador puede no poder reproducir ese resultado exacto. Por eso los benchmarks serios reportan promedios sobre multiples seeds.
- **Testing de aplicaciones**: tests unitarios que comparan output literal del modelo son fragiles. Hay que testear propiedades (formato, longitud, presencia de palabras clave), no strings exactos.
- **Confianza del usuario**: un usuario que pregunta "¿es seguro este medicamento?" dos veces puede recibir respuestas distintas. Eso erosiona la confianza, y es una de las razones por las que en dominios criticos (medicina, legal) se usa temperatura 0 o se hacen multiples queries y se vota.

---

## 5. Como hacerlo deterministico

Si necesitas reproducibilidad (debugging, testing, papers):

```python
# Antes de cualquier llamada al modelo:
torch.manual_seed(42)

# Si usas GPU:
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Algunas APIs (OpenAI, Anthropic, Gemini) ahora exponen un parametro `seed` precisamente para esto. Es un cambio reciente — hace dos anos no existia. La industria reconocio que la falta de reproducibilidad era un problema serio para uso profesional.

Importante: aun pasando la misma seed, **distintas versiones del modelo** o distinto hardware pueden producir resultados levemente distintos. La determinacion de bit-perfecto requiere fijar tambien version de framework, version de drivers, y arquitectura de GPU. Es por eso que los papers cuidadosos reportan "media ± stddev sobre 5 seeds" en lugar de un numero unico.

---

## 6. Las 5 generaciones tienen patrones distintos

Mira las salidas de la seccion 2:

- **seed 0**: tono melancolico ("I too have an them", "Whoping with thee"). El modelo arranca por una rama "introspectiva".
- **seed 42**: cita LEONTES (un personaje real de Winter's Tale), accion ("Had more he belood?"). Salta a un dialogo entre personajes.
- **seed 100**: tono filosofico ("Why, be say him well", "by heaves of me is count"). Soliloquio.
- **seed 1234**: tono dramatico ("offire of thy wretch", "puison"). Vocabulario violento.
- **seed 9999**: introduce a "FROMEY" (un personaje inventado), tono irreverente ("I am my geant"). Surreal.

Las 5 son **plausiblemente Shakespearianas**, pero distintas. Eso es exactamente lo que pasa con un LLM: las distribuciones aprendidas tienen muchos "modos" plausibles, y el sampling decide cual recorrer.

> El modelo aprendio una **distribucion sobre textos** que se parecen a Shakespeare. Cada seed elige un punto distinto de esa distribucion. No hay un unico Shakespeare "correcto" — hay un espacio de Shakespeares posibles, y el modelo se mueve por el.

---

## 7. La diferencia entre "creatividad" y "alucinacion"

Los LLMs son criticados por **alucinar** (generar info falsa). Pero la misma estocasticidad que les da creatividad les da alucinaciones — son la misma propiedad.

A temperatura/top-k bajos: poca variedad, pero tambien poca alucinacion. El modelo se queda en sus tokens mas seguros.

A temperatura/top-k altos: mas creativo, pero mas alucinaciones. El modelo se anima a tokens del rango medio que pueden ser brillantes o pueden ser invenciones.

{{< concept-alert type="recordar" >}}
No hay "creatividad sin riesgo de error" en LLMs. La estocasticidad es un cuchillo de doble filo: las mismas perillas (temperature, top-k, seed) que producen prosa interesante producen alucinaciones. Por eso los sistemas de produccion separan "modo creativo" (temperatura alta, para brainstorming) de "modo factual" (temperatura baja, para preguntas con respuesta verificable).
{{< /concept-alert >}}

Esto conecta con un debate vivo del campo: ¿se puede tener un LLM que **nunca** alucine? La respuesta corta: no, sin perder la habilidad de generar texto util. La generacion exige decidir entre tokens posibles, y cada decision puede equivocarse. Lo que se hace en la practica es bajar la tasa de alucinacion con tecnicas como **RAG** (recuperar fuentes y citarlas), **constrained decoding** (forzar al modelo a salir solo con palabras del corpus), o **verificacion post-hoc** (otro modelo revisa la respuesta).

---

## 8. Pausa de verificacion

1. ¿Por que el mismo prompt produce distintas respuestas si la temperatura y top-k son los mismos?
2. ¿Como haces que un LLM sea reproducible bit-a-bit?
3. ¿Cual es la conexion estructural entre creatividad y alucinacion?

---

## Cierre del viaje practico

Has visto:

- **8 escalones (01-08)**: construccion del Transformer desde cero. Embeddings, dot product, softmax, cross-entropy, autograd, gradient descent, Q/K/V, multi-head, bloque completo, mini-GPT entrenado.
- **7 experimentos (09-15)**: efectos de hyperparametros en el modelo entrenado. Profundidad, ancho, ratio FFN, contexto, temperatura, top-k, seed.

Total: ~16 scripts, ~1.5h de training acumulado, un Mini-GPT funcional, comprension end-to-end de como se entrenan los LLMs modernos.

Los proximos saltos serian:

- **Mas datos + mas compute** (scaling laws de Chinchilla y sucesores).
- **Fine-tuning con instrucciones** (RLHF, DPO, Constitutional AI).
- **Arquitecturas alternativas** (Mamba, RWKV, RetNet).
- **Multimodal** (CLIP, ViT, DiT).

Pero la base esta. **Ya entiendes los Transformers.** No conceptualmente — operacionalmente. Sabes que es una seed, que es un softmax, que es una causal mask, que es una cabeza de atencion. Sabes leer un paper de arquitectura y mapearlo a codigo. Sabes que hay detras de cada respuesta de ChatGPT que aparece en tu pantalla.

Eso no se desaprende.

---

Codigo: `clase_14/practica/12_seed_variety.py`

Volver al [hub de practica](..).
