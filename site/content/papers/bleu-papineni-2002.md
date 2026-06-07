---
title: "BLEU: Automatic Evaluation of Machine Translation"
weight: 243
math: true
---

{{< paper-card
    title="BLEU: a Method for Automatic Evaluation of Machine Translation"
    authors="Papineni, Roukos, Ward, Zhu"
    year="2002"
    venue="ACL 2002"
    pdf="/papers/bleu-papineni-2002.pdf" >}}
Introduce **BLEU** (Bilingual Evaluation Understudy), la métrica automática que durante más de veinte años fue el estándar de facto para evaluar traducción automática y, más tarde, *image captioning*. Su idea central es comparar la salida candidata contra varias referencias humanas mediante **modified n-gram precision** con *clipping* (que evita inflar la puntuación repitiendo palabras) combinada por media geométrica y corregida con una **brevity penalty** que castiga las traducciones demasiado cortas. El paper demuestra que BLEU **correlaciona 0.96–0.99 con el juicio humano** a nivel de corpus, siendo barata, rápida e independiente del idioma. Publicado en el ACL Anthology como **P02-1040**.
{{< /paper-card >}}

---

## Contexto

A comienzos de los 2000, la única forma confiable de evaluar un sistema de traducción automática (Machine Translation, MT) era someter su salida a **jueces humanos** que puntuaran adecuación y fluidez de cada oración. El paper resume el problema en tres frases: las evaluaciones humanas "son extensas pero caras", "pueden tomar meses en completarse" e "involucran trabajo humano que no puede reutilizarse".

Ese costo creaba un atasco (*logjam*): los desarrolladores necesitaban monitorear el efecto de cambios diarios en sus sistemas para descartar las malas ideas y quedarse con las buenas, pero no podían pagar una evaluación humana cada vez. La propuesta de Papineni y colegas es directa: hace falta una métrica automática **rápida, barata, independiente del idioma**, con costo marginal casi nulo por corrida y que **correlacione fuertemente con el juicio humano**. No para reemplazar a los jueces, sino para servir de suplente —un *understudy*— cuando se necesitan evaluaciones frecuentes. De ahí el nombre: **B**i**l**ingual **E**valuation **U**nderstudy.

El punto de vista que vertebra el método es una intuición simple: *cuanto más cerca esté una traducción automática de una traducción humana profesional, mejor es*. Operacionalizarla requiere una métrica numérica de "cercanía de traducción" y un corpus de referencias humanas de buena calidad. Esta métrica reaparece en la [Clase 23](/clases/clase-23) como herramienta para evaluar [Image Captioning](/fundamentos/image-captioning).

---

## Ideas principales

El paper parte de una observación que cualquier traductor reconoce: **hay muchas traducciones "perfectas" de una misma oración**. Pueden diferir en elección de palabras o en orden y aun así ser correctas. BLEU intenta capturar la capacidad humana de discriminar una buena traducción de una mala **contando coincidencias de n-gramas contra varias referencias**, independientes de la posición: cuantas más coincidencias, mejor el candidato.

### Modified n-gram precision con clipping

La piedra angular es la **precisión de n-gramas**. En su versión ingenua, la precisión de unigramas cuenta cuántas palabras del candidato aparecen en alguna referencia y divide por el total de palabras del candidato. El problema es que un sistema puede **sobregenerar** palabras razonables. El ejemplo clásico:

- **Candidato:** *the the the the the the the.*
- **Referencia 1:** *The cat is on the mat.* — **Referencia 2:** *There is a cat on the mat.*

La precisión ingenua daría $7/7 = 1.0$: puntuación perfecta para una traducción absurda. La solución es el **conteo recortado** (*clipping*): una palabra de la referencia se considera **agotada** una vez emparejada. Para cada n-grama se recorta su conteo en el candidato al máximo de veces que aparece en una sola referencia:

$$
\text{Count}_{\text{clip}} = \min(\text{Count},\ \text{Max\_Ref\_Count})
$$

Como *the* aparece a lo sumo dos veces en una referencia, $\text{Count}_{\text{clip}}(\text{the}) = \min(7, 2) = 2$, y la precisión de unigramas modificada cae a $p_1 = 2/7 \approx 0.286$. La fórmula general suma conteos recortados sobre todos los n-gramas de todas las oraciones candidatas del corpus y divide por el total de n-gramas candidatos (sin recortar):

$$
p_n = \frac{\sum_{\mathcal{C}} \sum_{\text{n-gram} \in \mathcal{C}} \text{Count}_{\text{clip}}(\text{n-gram})}{\sum_{\mathcal{C}'} \sum_{\text{n-gram}' \in \mathcal{C}'} \text{Count}(\text{n-gram}')}
$$

Los unigramas miden **adecuación** (¿están las palabras correctas?) y los n-gramas más largos miden **fluidez** (¿en orden gramatical razonable?). Como $p_n$ decae aproximadamente de forma exponencial con $n$, BLEU combina las precisiones con **media geométrica** (logaritmo con pesos uniformes), que da peso justo a cada escala y evita que las precisiones altas de $n$ bajo diluyan la señal de fluidez de los n-gramas largos. La media geométrica es dura: si **cualquier** $p_n = 0$ todo colapsa a cero, motivo del *smoothing* de Chen y Cherry (2014), ausente del paper original.

### Brevity penalty

El *clipping* penaliza candidatos demasiado **largos**, pero no los demasiado **cortos**: el candidato *of the* contra las referencias del ejemplo obtiene $p_1 = p_2 = 1.0$. El recall tradicional no resuelve esto con múltiples referencias (recordar todos los sinónimos a la vez —*I always invariably perpetually do*— produce una traducción peor que *I always do*). En su lugar, BLEU introduce un factor multiplicativo, la **brevity penalty**:

$$
BP =
\begin{cases}
1 & \text{si } c > r \\[4pt]
e^{\,1 - r/c} & \text{si } c \le r
\end{cases}
$$

donde $c$ es la longitud total del corpus de candidatos y $r$ es la **longitud efectiva de referencia** (suma, por oración, de la longitud de la referencia más cercana). Si el candidato es igual o más largo que la referencia, $BP = 1$; si es más corto, la penalización decae exponencialmente. La $BP$ se computa **sobre el corpus entero**, no oración por oración, para no castigar demasiado duro las desviaciones de longitud en oraciones cortas.

### La fórmula final

Reuniendo las piezas:

$$
BLEU = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

La línea base usa $N = 4$ y pesos uniformes $w_n = 1/4$, con *case folding* como única normalización. En dominio logarítmico, $\log BLEU = \min(1 - r/c,\ 0) + \sum_n w_n \log p_n$, donde el primer término es exactamente el logaritmo de la $BP$.

**Ejemplo numérico trabajado.** Supongamos $p_1 = 0.90$, $p_2 = 0.60$, $p_3 = 0.40$, $p_4 = 0.25$, con $c = 950$ y $r = 1000$.

*Media geométrica:*

$$
\tfrac14(\ln 0.90 + \ln 0.60 + \ln 0.40 + \ln 0.25) = \tfrac14(-2.9188) = -0.7297
$$
$$
\exp(-0.7297) = 0.4821
$$

*Brevity penalty* (como $c \le r$):

$$
BP = e^{\,1 - 1000/950} = e^{-0.0526} = 0.9488
$$

*BLEU final:* $BLEU = 0.9488 \cdot 0.4821 = \mathbf{0.457}$ (o 45.7 en escala 0–100). La brevedad leve (5% más corto) descuenta solo ~5%; el grueso lo determina la media geométrica, dominada por las precisiones bajas de los n-gramas largos.

### Corpus-level vs sentence-level

Un punto sutil y mal entendido: aunque la unidad básica de evaluación es la oración, BLEU se computa **acumulando numeradores y denominadores sobre todo el corpus**, no promediando BLEU por oración. Se recorren todas las candidatas sumando coincidencias recortadas y totales para cada $n$ antes de dividir, y se acumulan $c$ y $r$ para una única $BP$. La justificación es estadística: "BLEU solo necesita coincidir con el juicio humano cuando se promedia sobre un corpus; las puntuaciones de oraciones individuales suelen variar mucho". Calcular BLEU por oración y promediar da un número distinto y peor, porque cualquier $p_n = 0$ colapsa la oración a cero y la $BP$ por oración castiga las cortas demasiado duro. Por eso el "sentence-level BLEU" es ruidoso y requiere *smoothing*.

---

## Resultados experimentales

El paper valida lo único que importa de una métrica automática: **que prediga lo que dirían los humanos**. El setup comparó cinco "sistemas" —tres sistemas comerciales de MT (S1, S2, S3) y dos traductores humanos (H1 sin proficiencia nativa, H2 nativo)— sobre ~500 oraciones de noticias (chino→inglés) con hasta cuatro referencias, juzgadas por dos grupos de 10 personas (monolingüe y bilingüe), puntuando de 1 a 5.

| Sistema | S1 | S2 | S3 | H1 | H2 |
|---|---|---|---|---|---|
| BLEU | 0.0527 | 0.0829 | 0.0930 | 0.1934 | 0.2571 |

El orden BLEU (S1 < S2 < S3 < H1 < H2) coincide con el orden de calidad humano. Nótese que **ni los traductores humanos obtienen 1.0**: como hay muchas traducciones válidas, ni un humano coincide exactamente con las referencias. (A más referencias, mayor BLEU: un humano sacó 0.3468 con cuatro referencias pero solo 0.2571 con dos, por lo que comparar BLEU con distinto número de referencias es engañoso.)

Dividiendo el corpus en 20 bloques y corriendo t-tests pareados entre sistemas adyacentes, todas las diferencias resultaron estadísticamente muy significativas (t ≥ 3.4 ≫ 1.7), incluido el delicado par S2 vs. S3. El resultado estrella: la regresión lineal de las puntuaciones humanas en función de BLEU dio **coeficiente de correlación 0.99** para el grupo monolingüe y **0.96** para el bilingüe. BLEU rastrea el juicio humano muy de cerca y distingue correctamente sistemas de calidad parecida.

---

## BLEU en Image Captioning

Aunque nació para MT, el paper ya anticipaba su generalización a "*summarization* u otras tareas de NLG similares". La adaptación a [Image Captioning](/fundamentos/image-captioning) es casi mecánica: no hay oración fuente en otro idioma, pero sí el ingrediente esencial, **múltiples referencias humanas**. En **MS COCO Captions** cada imagen tiene **5 captions humanos** que juegan el papel de las referencias de traducción; el caption generado por el modelo es el "candidato".

Los papers de *captioning* reportan típicamente **BLEU-1, BLEU-2, BLEU-3 y BLEU-4** (BLEU con $N = 1, 2, 3, 4$). BLEU-1 mide cobertura de vocabulario; BLEU-4 exige coincidencias de frases de cuatro palabras y es mucho más exigente con la fluidez. Es habitual ver las cuatro cifras juntas, como en [Bottom-Up/Top-Down Attention](/papers/bottom-up-attention-anderson-2018) (BLEU-4 36.2). Una vez generado el caption con una estrategia de decodificación (*greedy* o *beam search*, ver [decoding strategies](/fundamentos/decoding-strategies)), BLEU lo **evalúa** contra los captions de referencia.

En *captioning* las debilidades de BLEU son aún más agudas: premia repetir palabras frecuentes (un modelo puede subir BLEU produciendo captions genéricos y seguros), no mide relevancia visual (compara cadenas, no tiene acceso a la imagen) y correlaciona mal a nivel de imagen individual. Por eso la comunidad desarrolló métricas específicas: **CIDEr** (Vedantam 2015) pondera n-gramas por TF-IDF para premiar lo distintivo sobre lo genérico —el agujero exacto de BLEU en COCO— y **SPICE** (Anderson 2016) compara grafos de escena (objetos, atributos, relaciones). En la práctica se reporta el cuarteto BLEU-1..4 junto con METEOR, CIDEr y SPICE, porque ninguna métrica sola captura todo. Estos temas pertenecen al [dominio Multimodal](/dominios/multimodal).

---

## Limitaciones

BLEU mide **similitud superficial de cadenas**, y ese es su techo:

- **No captura significado ni semántica.** Solo cuenta coincidencias de n-gramas; no tiene noción de sinónimos ni de roles semánticos.
- **Penaliza paráfrasis válidas.** Decir *automobile* donde la referencia dice *car* recibe precisión baja injustamente. Más referencias mitigan, pero siempre son finitas.
- **Sensible a la tokenización.** Decisiones de tokenización cambian la puntuación, lo que hizo que durante años las cifras BLEU no fueran comparables entre papers, hasta que **SacreBLEU** (Post, 2018) estandarizó la tokenización.
- **Problemática a nivel de oración individual.** Diseñada para corpus; por oración es ruidosa y colapsa a cero ante cualquier $p_n = 0$.
- **No mide orden global ni coherencia de largo alcance.** Captura fluidez local hasta $N = 4$, pero no estructura del discurso.

Estas grietas motivaron las métricas posteriores: **METEOR** (Banerjee y Lavie, 2005) añade *stemming*, sinónimos vía WordNet y recall explícito; **BERTScore** (Zhang et al., 2020) reemplaza el conteo de n-gramas por similitud coseno de embeddings contextuales; **BLEURT** (Sellam et al., 2020) entrena un modelo para predecir directamente el juicio humano. Todas atacan el mismo punto: BLEU mide superficie, no significado.

---

## Por qué importa hoy

BLEU es una de las métricas más influyentes de la historia del NLP. Durante más de veinte años fue la métrica de facto para reportar resultados en traducción automática y *image captioning*, sustentada en tres propiedades: es **barata**, **rápida** y **suficientemente correlacionada** con el juicio humano para guiar la iteración de investigación.

Su legado va más allá del uso directo: **definió el paradigma** de evaluación automática de NLG por comparación con referencias humanas. La idea de precisión de n-gramas recortada, combinada con media geométrica y penalizada por brevedad, reaparece transformada en casi toda métrica posterior. [ROUGE (Lin 2004)](/papers/rouge-lin-2004) es esencialmente "BLEU pero recall-oriented" para *summarization*; METEOR, NIST, CIDEr y chrF son variaciones sobre el tema que BLEU inauguró.

Incluso en la era de los LLM, BLEU sigue vivo: aparece en *benchmarks* de traducción, se usa como señal barata en *ablations* y es el punto de comparación histórico obligatorio. Aunque para evaluación de calidad fina hoy se prefieren métricas neuronales (COMET, BLEURT, BERTScore) o *LLM-as-a-judge*, BLEU permanece como la línea base universal.

---

## Notas y enlaces

- **Relación con ROUGE.** BLEU normaliza por el **candidato** (precision: ¿lo que generé es correcto?); [ROUGE](/papers/rouge-lin-2004) normaliza por la **referencia** (recall: ¿cubrí el contenido?). BLEU domina en traducción y *captioning*; ROUGE domina en *summarization*. Ambas comparten raíz: conteo de n-gramas contra referencias humanas validado por correlación con jueces. Ver [fundamento BLEU](/fundamentos/bleu-metric) y [fundamento ROUGE](/fundamentos/rouge-metric).
- **SacreBLEU** (Post, 2018): no es una métrica nueva sino una implementación estandarizada de BLEU con tokenización fija, que resolvió la comparabilidad entre papers. Si reportas BLEU hoy, usa SacreBLEU.
- **Detalle de implementación heredado del paper:** $N = 4$, pesos uniformes $w_n = 1/4$, *case folding* como única normalización y cómputo a nivel de **corpus** (no de oración). El *smoothing* para el caso $p_n = 0$ (Chen y Cherry, 2014) es un añadido posterior.
- **Referencia primaria:** Papineni, K., Roukos, S., Ward, T. y Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL 2002, pp. 311–318. ACL Anthology P02-1040. <https://aclanthology.org/P02-1040/>
