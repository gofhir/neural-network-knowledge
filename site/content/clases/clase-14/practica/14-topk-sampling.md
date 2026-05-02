---
title: "14 - Top-k sampling: estrategias de generacion"
weight: 140
math: true
---

En el escalon 08 entrenamos al mini-GPT y lo vimos generar Shakespeare. Pero hicimos algo medio escondido: en cada paso de generacion sampleabamos del vocabulario completo con `torch.multinomial(probs, num_samples=1)`. Eso funciono, pero esta lejos de ser la estrategia que usan los LLMs reales. ChatGPT, Claude, Gemini, LLaMA — todos usan variantes mas sofisticadas: **top-k**, **top-p**, **temperature**, o combinaciones. Este capitulo es para entender por que existen, que problema resuelven y como afectan la salida del modelo.

El script que acompana es `clase_14/practica/11_topk_sampling.py`. Entrena el mismo mini-GPT y produce comparaciones lado a lado de cinco estrategias de sampling sobre el mismo prompt.

---

## 1. El problema del sampling

El modelo, despues de un forward pass, te entrega una distribucion de probabilidad sobre los 65 caracteres posibles. La pregunta es: **¿como elegis el siguiente token de esa distribucion?**

La respuesta naive — "elige cualquiera de acuerdo a su probabilidad" — es lo que hace `torch.multinomial`. Y tiene problemas reales:

- **A veces samplea tokens muy improbables.** Aunque la probabilidad sea 0.01%, hay 1 chance entre 10000 de que salga. En cientos de pasos de generacion, eso pasa varias veces. Y cuando pasa, el modelo "se desvia" hacia palabras inventadas, errores tipograficos, o transiciones bruscas.
- **A veces el modelo "confia mucho" en una sola opcion** — la probabilidad de un token domina (digamos 95%). Sampleas casi siempre ese mismo token, y el modelo cae en bucles repetitivos.

Lo que necesitamos es una manera de balancear **diversidad y calidad**. Que el modelo varie su salida (no caiga en loops) pero que no diga locuras (no genere tokens absurdos).

Las estrategias que vienen a continuacion atacan exactamente ese trade-off.

---

## 2. Las cuatro estrategias

### 2.1 Greedy (top_k=1)

La estrategia mas simple posible: en cada paso, elegir el token con probabilidad mas alta.

$$
\text{token} = \arg\max_i P_i
$$

Sin variedad. Es completamente deterministico — dado el mismo prompt, **siempre** genera la misma salida. Caso extremo: te perdiste todo el resto de la distribucion. Suele caer en loops, como veremos en los resultados.

### 2.2 Multinomial libre (sin filtrado)

La estrategia que usamos en el escalon 08: samplear de toda la distribucion. Cada uno de los 65 caracteres tiene chance proporcional a su probabilidad, **incluyendo los muy improbables**. Maxima variedad, pero abierta a tokens raros y errores tipograficos.

### 2.3 Top-k

Idea: solo considerar los **k tokens mas probables**, ignorar el resto, y samplear entre esos k. El procedimiento:

1. Tomar los $k$ tokens con mayor probabilidad.
2. Renormalizar sus probabilidades (que vuelvan a sumar 1).
3. Samplear segun esa distribucion truncada.

Si $k = 5$, descartas todos los tokens "raros" pero mantienes variedad entre los 5 mas plausibles. Es el balance perfecto: diverso, sin caer en lo absurdo.

### 2.4 Temperature

Esta es **ortogonal** al top-k — no es una alternativa, es un parametro complementario. Modifica que tan "pico" es la distribucion **antes** de samplear:

$$
P_i \propto \exp(\text{logit}_i / T)
$$

- $T < 1$: la distribucion se vuelve mas afilada (los logits altos se vuelven aun mas dominantes). Mas determinista.
- $T > 1$: la distribucion se vuelve mas plana (todos los tokens se acercan en probabilidad). Mas creativo / aleatorio.
- $T = 1$: distribucion sin modificar.

Combinada con top-k da control fino: top-k=5 + temp=0.8 significa "elige entre los 5 mas plausibles, pero con preferencia clara hacia el mas probable".

---

## 3. La implementacion

Asi luce la funcion `generate` con soporte para top-k y temperature:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k: solo considerar los k mas probables
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Tres lineas merecen analisis:

- `logits = logits[:, -1, :] / temperature` — solo nos importa la **ultima** posicion (es la que predice el siguiente token). Dividir por la temperatura **antes** del softmax es lo que afila/aplana la distribucion.

- `v, _ = torch.topk(logits, min(top_k, logits.size(-1)))` — `torch.topk` devuelve los $k$ valores mas grandes y sus indices. `v` queda con shape `(batch, k)` con los logits mas altos.

- `logits[logits < v[:, [-1]]] = -float('inf')` — esta es la **linea clave**. `v[:, [-1]]` agarra el k-esimo logit mas alto (el "umbral" del top-k). Cualquier logit por debajo se reemplaza por $-\infty$. Cuando despues hagamos softmax, $e^{-\infty} = 0$ — esos tokens van a tener probabilidad 0, y nunca seran sampleados. Mismo truco que usamos en la causal mask del escalon 08.

El indexing `[:, [-1]]` (con corchetes dobles) en lugar de `[:, -1]` es para mantener la dimension y permitir broadcasting con `logits`. Si fuera `[:, -1]`, perderias la dimension de batch y la comparacion no funcionaria igual.

---

## 4. Resultados reales (del script)

Mismo modelo entrenado por 3000 iteraciones, mismo prompt `'ROMEO:'`, mismo seed (`torch.manual_seed(42)`), pero variando la estrategia de sampling. Asi sentis directo el efecto de cada parametro.

### 4.1 Greedy (k=1)

```
ROMEO:
I have the shall shall the sentle of the cousint
That the shall so be the soul of the courted
To the court of the court of the courtions of the court
That the shall so be the souls of the courts,
```

**Loop catastrofico**: "the court of the court of the courtions of the court". Cero variedad. El modelo eligio "the" como mas probable, llego a un estado donde despues de "the" lo mas probable era "court", despues de "court" lo mas probable era "of", despues de "of" lo mas probable era "the", y vuelve a empezar.

Es el problema clasico del greedy decoding: como nunca "exploras", caes en un atractor del modelo.

### 4.2 Multinomial libre (sin filtrado)

```
ROMEO:
-royal, kill yet, and commal cal maid too,
But me solding a kit you, undranged hunder blang,
For she had forgest is of your more over negge exlapues.
Savy, sharm, tall tongraix aspely a glowabide,
```

**Variado pero con errores**: aparecen palabras inventadas como `glowabide`, `tongraix`, `exlapues`, `hunder blang`. La cadencia es variada — no hay loops — pero el modelo a veces samplea tokens con probabilidad muy baja y el resultado se desordena.

Notese: muchas palabras tipo "Savy", "sharm" tienen apariencia de ingles pero no existen. El modelo, sin filtrado, "explora" el vocabulario completo y a veces eso lo lleva a callejones sin salida fonetica.

### 4.3 Top-k=5

```
ROMEO:
Troy, make the chargue, some of our hole;
Then stand stay a shope, my horget strung to speed:
I'll to made my hears, they commen out the bace
to tean'd housbalt so too tont: a bastel as thee
With, and
```

**Balance**: coherente pero diverso. Sin loops, sin palabras totalmente inventadas. Aparecen palabras casi correctas ("chargue" cerca de "charge", "shope" cerca de "shape"), contracciones ("I'll", "tean'd"), vocabulario shakespeariano ("thee").

El modelo solo elige entre los 5 mas plausibles en cada paso. Eso lo mantiene en territorio fonetico legitimo, pero permite suficiente variedad para no repetirse.

### 4.4 Top-k=20

```
ROMEO:
Troy, make op
the approuse think made to to calm your their king,
Shall way fear bene blang, and strouces of by,
Be offighter all him negge exlapues his moy,
```

Mas variedad que k=5. Aparecen mas palabras "raras" ("approuse", "strouces", "offighter", "exlapues"), porque ahora el modelo puede elegir entre 20 tokens en vez de 5. La salida es mas creativa pero tambien empieza a sonar menos coherente que con k=5.

Hay un **trade-off**: $k$ mas chico = mas conservador, mas pegado a las predicciones del modelo. $k$ mas grande = mas variedad, mas riesgo de "desvio".

### 4.5 Top-k=5 + temperature=0.8

```
ROMEO:
Troy, make the chargue, some of that with them of her their his
Is parn thy bear of the servingess of mades of to souch,
Where with all hear highness, any soul.

CAMILLO:
I am there that with a do the
```

**Mejor resultado**. Top-k=5 mantiene el filtro de calidad (sin tokens absurdos), y temperature=0.8 afila ligeramente la distribucion para favorecer los tokens mas probables dentro de esos 5.

Lo mas notable: aparece **CAMILLO**. Camillo es un personaje real de *The Winter's Tale* de Shakespeare. El modelo, con esta combinacion, no solo genera frases coherentes en estilo Shakespeare sino que **transiciona correctamente entre personajes**. Best of both worlds: diversidad controlada + temperatura que favorece coherencia.

{{< concept-alert type="clave" >}}
La combinacion **top-k + temperature < 1** es la formula mas usada en LLMs comerciales. Top-k filtra "loquras", temperature concentra masa en lo plausible. Solos cada uno tiene defectos; juntos se complementan.
{{< /concept-alert >}}

---

## 5. Por que greedy falla

Tomemos un momento para entender por que greedy cae en loops, porque la intuicion es importante.

Greedy es deterministico: en cada paso elige el token con maxima probabilidad. Pero la "decision optima local" no es la "decision optima global". Si en cada paso elige el token mas probable, y resulta que despues de cierta secuencia el token mas probable es uno que ya genero, vas a repetir ese ciclo **siempre**, porque el modelo es deterministico y entrara al mismo estado.

En el ejemplo: el modelo aprende durante entrenamiento que despues de "the" suele venir "court", despues de "court" suele venir "of", despues de "of" suele venir "the". Con sampling estocastico esto se rompe: ocasionalmente eliges otra cosa y "escapas". Pero greedy no — siempre elige el mismo, y queda atrapado.

Esto es un problema teorico bien conocido: para distribuciones de lenguaje natural, la moda no genera secuencias representativas. Generar texto "natural" requiere algo de aleatoriedad. **Greedy NO se usa en LLMs comerciales** por esta razon. Siempre hay algo de estocasticidad para mantener la generacion fluida.

---

## 6. Top-p (nucleus) sampling

Top-k tiene una limitacion: el numero de tokens que considera es **fijo**. Pero la cantidad "razonable" de candidatos depende del contexto. A veces el modelo esta muy seguro (1-2 tokens dominan), a veces hay 10 tokens igualmente plausibles. Top-k no se adapta a eso.

**Top-p** (tambien llamado **nucleus sampling**) ataca exactamente esta limitacion. En vez de "los k mas probables", define "los tokens cuya probabilidad acumulada llega a $p$". Procedimiento:

1. Ordenar los tokens por probabilidad descendente: $P_1 \geq P_2 \geq \dots$
2. Acumular las probabilidades: $P_1$, $P_1 + P_2$, $P_1 + P_2 + P_3$, ...
3. Cortar cuando la suma acumulada llega a $p$ (ejemplo: $p = 0.9$).
4. Samplear de los tokens dentro del corte.

Si $p = 0.9$ y el modelo esta muy seguro (la moda concentra el 95% de la masa), el "nucleo" puede ser de 1-2 tokens. Si el modelo es inseguro y la masa esta repartida, el "nucleo" puede ser de 30+ tokens.

**Ventaja sobre top-k**: el "k efectivo" se adapta dinamicamente a la confianza del modelo. Mas robusto a contextos heterogeneos.

LLMs comerciales suelen combinar las tres: **top-k=40, top-p=0.9, temperature=0.7**. Cada parametro corta una zona del espacio de tokens:

- Temperature: afila/aplana la distribucion.
- Top-k: corta numero maximo de candidatos.
- Top-p: corta candidatos cuya probabilidad acumulada exceda $p$.

---

## 7. Settings comunes en LLMs

Aproximadamente estos son los defaults publicos o documentados:

| Modelo                  | Defaults tipicos                              |
|-------------------------|-----------------------------------------------|
| ChatGPT (web)           | temp=0.7, top_p=1.0                           |
| Claude (web)            | temp~1.0, top_p=0.99                          |
| GPT-4 API               | temp=1.0, top_p=1.0 (defaults), pero hay control |
| LLaMA inference         | temp=0.6-0.8, top_p=0.9-0.95, top_k=40        |

Las APIs te dan control sobre estos parametros. Mas temperatura / top_p mas alto = mas variedad y "creatividad". Menos = mas conservador y reproducible. Cada use case tiene su sweet spot:

- **Codigo / razonamiento matematico**: temperature baja (0.0-0.3). Quieres respuestas deterministicas, precisas, repetibles.
- **Conversacion casual**: temperature media (0.7-1.0). Quieres respuestas naturales, variadas.
- **Escritura creativa / brainstorming**: temperature alta (1.0-1.3). Quieres variedad, exploracion, sorpresa.

Cuando ves "modo creativo" vs "modo preciso" en una UI, generalmente esos dos botones son shortcuts que ajustan estos parametros.

---

## 8. Conexion con el modelo entrenado

Lo importante de este capitulo es que el modelo es el mismo. Mismo `mini_gpt`, mismos pesos, misma cantidad de iteraciones de entrenamiento. La unica diferencia es **como** sampleamos de las distribuciones de salida.

Eso vale la pena meditar: el comportamiento aparente de un LLM depende no solo de su entrenamiento, sino tambien de la **estrategia de decoding**. Dos LLMs con el mismo modelo pueden producir salidas radicalmente distintas si una usa greedy y la otra usa top-k=40 + temperature=0.7.

{{< concept-alert type="recordar" >}}
Cuando un LLM "responde mal" o "se vuelve repetitivo", muchas veces no es un problema del modelo sino de los parametros de sampling. Probar variar `temperature`, `top_k`, `top_p` antes de concluir que el modelo no sabe algo es una intuicion barata y poderosa.
{{< /concept-alert >}}

---

## 9. Pausa de verificacion

Antes de pasar al siguiente escalon, asegurate de poder responder con tus propias palabras:

1. **¿Por que greedy cae en loops?** (Pista: deterministico + estado del modelo + atractores.)
2. **¿Cual es la diferencia entre top-k y top-p?** (Pista: numero fijo vs masa de probabilidad acumulada.)
3. **¿Para que sirve la temperatura combinada con top-k?** (Pista: top-k filtra "que tokens", temperatura modula "cuanto se diferencia el mas probable de los demas".)
4. **¿Por que el modelo con top-k=5 + temp=0.8 logro generar "CAMILLO" como personaje siguiente?** (Pista: el filtrado mantiene calidad, la temperatura concentra masa en transiciones plausibles aprendidas.)

---

## 10. Que viene despues

Ya tienes el modelo entrenado y entendes como modular su salida. En el siguiente capitulo vamos a explorar **el efecto del seed**: misma estrategia de sampling, distintos seeds, distintos textos generados. Esa es la otra fuente de variedad — la estocasticidad del PRNG mismo.

---

Codigo: `clase_14/practica/11_topk_sampling.py`

Siguiente: [15 - Variedad con seeds](../15-seed-variety).
