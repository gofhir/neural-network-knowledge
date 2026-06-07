---
title: "BLEU: a Method for Automatic Evaluation of Machine Translation"
authors: "Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu"
venue: "Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL 2002), Philadelphia, July 2002, pp. 311–318"
year: 2002
anthology_id: "P02-1040"
link: "https://aclanthology.org/P02-1040/"
math: true
---

# BLEU: a Method for Automatic Evaluation of Machine Translation

> **Cita:** Kishore Papineni, Salim Roukos, Todd Ward y Wei-Jing Zhu. *BLEU: a Method for Automatic Evaluation of Machine Translation*. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, julio de 2002, pp. 311–318. ACL Anthology ID: **P02-1040**. IBM T. J. Watson Research Center, Yorktown Heights, NY. Enlace: <https://aclanthology.org/P02-1040/>

Este análisis es el equivalente para **BLEU** del fundamento `rouge-metric.md` que ya existe en el site. BLEU nació en 2002 para evaluar **traducción automática** (Machine Translation, MT), pero se transformó en la métrica de facto para casi toda tarea de **generación de lenguaje natural** que compara una salida contra referencias humanas: traducción, *image captioning*, *summarization*, simplificación y *headline generation*. En la clase 23 (slide 27) aparece como la métrica para evaluar **Image Captioning**: "puntúa de 0 a 1, se centra en la similitud de cadenas, no evalúa la calidad de la traducción". Esta última frase, que parece una crítica, es en realidad una cita casi literal de la filosofía del paper: BLEU no entiende nada de significado; solo mide cuánto se **parece superficialmente** un candidato a una o varias traducciones humanas profesionales.

---

## 1. Contexto: el cuello de botella de evaluar MT en 2002

A comienzos de los 2000, la evaluación de un sistema de traducción automática era un problema operativo serio. La única forma confiable de saber si un sistema traducía bien era **someter su salida a jueces humanos** que puntuaran adecuación (*adequacy*), fidelidad (*fidelity*) y fluidez (*fluency*) de cada oración. Los autores resumen el problema en tres frases del *abstract*: las evaluaciones humanas "son extensas pero caras", "pueden tomar meses en completarse" e "involucran trabajo humano que no puede reutilizarse".

Ese costo creaba lo que el paper llama un **logjam** (un atasco): los desarrolladores de MT necesitaban monitorear el efecto de cambios diarios en sus sistemas para descartar ideas malas y quedarse con las buenas, pero no podían pagar una evaluación humana cada vez que tocaban una línea de código. La consecuencia era que muchas ideas de investigación prometedoras se quedaban esperando a ser liberadas del cuello de botella de la evaluación.

La propuesta de Papineni y colegas es directa: necesitamos una métrica automática que sea **rápida, barata, independiente del idioma**, con un **costo marginal casi nulo por corrida** y, lo más importante, que **correlacione fuertemente con el juicio humano**. La idea no es reemplazar a los jueces humanos para siempre, sino construir un *understudy* —un suplente, un doble de acción— que los sustituya cuando se necesitan evaluaciones rápidas o frecuentes. De ahí el nombre, explicado en una nota al pie de la primera página: **B**i**l**ingual **E**valuation **U**nderstudy.

### 1.1. El punto de vista (*viewpoint*)

La sección 1.2 del paper formula la pregunta central: *¿cómo se mide el desempeño de una traducción?* La respuesta que adoptan, y que vertebra todo el método, es:

> Cuanto más cerca esté una traducción automática de una traducción humana profesional, mejor es.

Para operacionalizar esa intuición hacen falta dos ingredientes:

1. Una **métrica numérica de "cercanía de traducción"** (*translation closeness*).
2. Un **corpus de traducciones humanas de referencia** de buena calidad.

La métrica de cercanía la modelaron a partir de la *word error rate* (tasa de error por palabra) usada por la comunidad de reconocimiento de voz, pero modificándola para admitir **múltiples referencias** y para tolerar **diferencias legítimas en la elección de palabras y el orden**. La idea final es un promedio ponderado de coincidencias de frases (n-gramas) de longitud variable contra las referencias.

---

## 2. La idea central: comparar n-gramas contra referencias

El paper parte de una observación que cualquier traductor reconoce: **hay muchas traducciones "perfectas" de una misma oración fuente**. Pueden diferir en la elección de palabras o en el orden, y aun así un humano distingue sin esfuerzo una buena traducción de una mala. BLEU intenta capturar esa capacidad de discriminación.

Consideremos el ejemplo del paper (Example 1), dos candidatos de traducción de una oración fuente en chino:

- **Candidato 1:** *It is a guide to action which ensures that the military always obeys the commands of the party.*
- **Candidato 2:** *It is to insure the troops forever hearing the activity guidebook that party direct.*

Ambos hablan del mismo tema, pero difieren marcadamente en calidad. Para cuantificarlo, el paper provee **tres referencias humanas**:

- **Referencia 1:** *It is a guide to action that ensures that the military will forever heed Party commands.*
- **Referencia 2:** *It is the guiding principle which guarantees the military forces always being under the command of the Party.*
- **Referencia 3:** *It is the practical guide for the army always to heed the directions of the party.*

El Candidato 1 comparte muchas palabras y frases con las tres referencias ("It is a guide to action", "which", "ensures that the", "military", "always", "commands", "of the party"). El Candidato 2 exhibe muchas menos coincidencias y más cortas. La tarea programática primaria de quien implementa BLEU es exactamente esa: **comparar los n-gramas del candidato con los n-gramas de las referencias y contar las coincidencias**. Las coincidencias son independientes de la posición. Cuantas más coincidencias, mejor el candidato.

---

## 3. *Modified n-gram precision*: el corazón de BLEU

### 3.1. Por qué la precisión ingenua se rompe

La piedra angular de BLEU es la **precisión de n-gramas**. En su versión ingenua, la precisión de unigramas es: *cuente cuántas palabras del candidato aparecen en alguna referencia, y divida por el total de palabras del candidato.*

El problema es que los sistemas de MT pueden **sobregenerar** palabras "razonables", produciendo traducciones improbables pero de altísima precisión. El ejemplo (Example 2) es el clásico que aparece en toda clase de NLP:

- **Candidato:** *the the the the the the the.*
- **Referencia 1:** *The cat is on the mat.*
- **Referencia 2:** *There is a cat on the mat.*

La palabra *the* aparece en las referencias, así que las 7 ocurrencias del candidato "cuentan". La precisión de unigramas ingenua sería $7/7 = 1.0$: una puntuación perfecta para una traducción absurda. La métrica está claramente rota.

### 3.2. El *clipping*: la modificación clave

La solución del paper es el **conteo recortado** (*clipping*). La intuición que formalizan: una palabra de la referencia debe considerarse **agotada** una vez que ya fue emparejada por una palabra del candidato. Operativamente:

1. Para cada n-grama del candidato, calcula su conteo en el candidato, $\text{Count}$.
2. Calcula el **máximo número de veces** que ese n-grama aparece en **una sola** referencia. Llámalo $\text{Max\_Ref\_Count}$.
3. Recorta el conteo del candidato a ese máximo:

$$
\text{Count}_{\text{clip}} = \min(\text{Count},\ \text{Max\_Ref\_Count})
$$

4. Suma los conteos recortados de todos los n-gramas únicos del candidato y divide por el **total de n-gramas del candidato** (sin recortar).

Para el ejemplo de *the the the…*: en la Referencia 1 aparece *the* dos veces (*The* y *the*), en la Referencia 2 una vez. El máximo en una sola referencia es 2. Entonces $\text{Count}_{\text{clip}}(\text{the}) = \min(7, 2) = 2$. La **precisión de unigramas modificada** es:

$$
p_1 = \frac{2}{7} \approx 0.286
$$

frente al $7/7$ de la versión ingenua. El absurdo queda penalizado.

### 3.3. La fórmula general

Para un n-grama de cualquier longitud, y sobre un corpus completo de candidatos, la precisión de n-gramas modificada es:

$$
p_n = \frac{\displaystyle\sum_{\mathcal{C}\,\in\,\{\text{Candidates}\}} \ \sum_{\text{n-gram}\,\in\,\mathcal{C}} \text{Count}_{\text{clip}}(\text{n-gram})}{\displaystyle\sum_{\mathcal{C}'\,\in\,\{\text{Candidates}\}} \ \sum_{\text{n-gram}'\,\in\,\mathcal{C}'} \text{Count}(\text{n-gram}')}
$$

donde el numerador acumula conteos **recortados** sobre todos los n-gramas de todas las oraciones candidatas del corpus, y el denominador acumula los conteos **sin recortar** (el total de n-gramas candidatos).

### 3.4. Ejemplo numérico trabajado del propio paper

El paper reporta las precisiones modificadas de los candidatos del Example 1:

| Candidato | $p_1$ (unigramas) | $p_2$ (bigramas) |
|-----------|-------------------|------------------|
| Candidato 1 (bueno) | $17/18 \approx 0.944$ | $10/17 \approx 0.588$ |
| Candidato 2 (malo) | $8/14 \approx 0.571$ | $1/13 \approx 0.077$ |

El contraste es nítido: el buen candidato tiene casi todos sus unigramas y más de la mitad de sus bigramas emparejados; el candidato malo cae a la mitad en unigramas y se desploma en bigramas. El paper observa que esto captura **dos aspectos** de la traducción simultáneamente: los unigramas (1-gramas) miden **adecuación** (¿están las palabras correctas?), mientras que los n-gramas más largos miden **fluidez** (¿están en un orden gramatical razonable?).

### 3.5. ¿Sirve la precisión modificada para rankear sistemas?

La sección 2.1.2 del paper valida empíricamente que la precisión modificada **distingue traducciones buenas de malas**. Calcularon las precisiones sobre la salida de un buen traductor humano y un sistema de MT pobre, usando 4 referencias para cada una de 127 oraciones fuente. El resultado (Figura 1 del paper) es contundente: el humano alcanza precisiones mucho más altas que la máquina en las cuatro longitudes, y la **brecha se ensancha** a medida que $n$ crece de 1 a 4. La señal es tan fuerte que cualquier precisión individual basta para separar humano de máquina.

Más exigente es distinguir entre traducciones de calidad **parecida**. En la Figura 2 compararon cinco "sistemas" (dos humanos H1, H2 y tres sistemas comerciales S1, S2, S3) contra dos referencias. La precisión modificada los ordenó como H2 > H1 > S3 > S2 > S1 —exactamente el mismo orden que luego asignarían los jueces humanos.

---

## 4. Combinar las precisiones: por qué media geométrica

Hasta aquí tenemos cuatro números separados $p_1, p_2, p_3, p_4$. ¿Cómo combinarlos en una sola cifra?

La clave es una observación empírica de la Figura 2: **la precisión modificada decae aproximadamente de forma exponencial con $n$**. La precisión de unigramas es mucho mayor que la de bigramas, que a su vez es mucho mayor que la de trigramas, etcétera. Esto tiene sentido: es fácil acertar palabras sueltas, difícil acertar secuencias largas idénticas.

Si usáramos un **promedio aritmético** $\frac{1}{4}(p_1+p_2+p_3+p_4)$, las precisiones grandes de $n$ bajo dominarían y las diferencias en los n-gramas largos —justo donde vive la señal de fluidez— quedarían diluidas. Un esquema de promediado razonable debe **tener en cuenta este decaimiento exponencial**. La respuesta del paper es promediar **el logaritmo** de las precisiones con pesos uniformes, lo que equivale a tomar la **media geométrica**:

$$
\left(\prod_{n=1}^{N} p_n\right)^{1/N} = \exp\!\left(\frac{1}{N}\sum_{n=1}^{N} \log p_n\right)
$$

El logaritmo convierte el decaimiento multiplicativo en algo aditivo y tratable, y la media geométrica da un peso relativo justo a cada escala. El paper reporta dos hallazgos en notas al pie: (1) usar la media geométrica **correlaciona algo mejor** con el juicio humano que la aritmética; (2) la media geométrica es **dura** —si **cualquier** $p_n$ se hace cero, todo el producto colapsa a cero— pero argumentan que en corpus de tamaño razonable con $N_{\max} \le 4$ eso es un evento extremadamente raro. (Es exactamente esta dureza la que motiva, en implementaciones modernas, el *smoothing* de Chen y Cherry 2014, ausente del paper original.)

---

## 5. *Brevity Penalty* (BP): el agujero del recall

### 5.1. El problema: la precisión no castiga candidatos cortos

La precisión de n-gramas modificada ya penaliza candidatos demasiado **largos** (palabras espurias que no aparecen en ninguna referencia bajan la precisión, y el *clipping* castiga repetir una palabra más veces de las que aparece en las referencias). Pero **falla en el otro extremo**: no castiga candidatos demasiado **cortos**. El ejemplo (Example 3) lo muestra:

- **Candidato:** *of the*
- contra las mismas tres referencias del Example 1.

Como el candidato es minúsculo, sus precisiones se **inflan**: precisión de unigramas $= 2/2 = 1.0$ y precisión de bigramas $= 1/1 = 1.0$. Una traducción de dos palabras obtiene precisión perfecta. Esto es exactamente el problema simétrico al de *the the the*.

### 5.2. Por qué no se usa recall

La forma tradicional de tapar este agujero sería **emparejar precisión con recall**. Pero el paper argumenta en la sección 2.2.1 ("The trouble with recall") que el recall **no funciona** cuando hay múltiples referencias. El ejemplo (Example 4):

- **Candidato 1:** *I always invariably perpetually do.*
- **Candidato 2:** *I always do.*
- **Referencia 1:** *I always do.* — **Referencia 2:** *I invariably do.* — **Referencia 3:** *I perpetually do.*

El Candidato 1 "recuerda" más palabras de las referencias (*always*, *invariably*, *perpetually*), pero es obviamente **peor** que el Candidato 2. Recordar **todas** las opciones sinónimas a la vez produce una traducción mala. Por lo tanto, un recall ingenuo sobre el conjunto de todas las palabras de referencia es una medida inadecuada, y calcular recall sobre conceptos (alineando sinónimos) es demasiado complicado dado que las referencias varían en longitud y sintaxis.

### 5.3. La solución: penalización por brevedad

En lugar de recall, BLEU introduce un **factor multiplicativo de penalización por brevedad** (*brevity penalty*, BP). Las traducciones más **largas** que la referencia ya están castigadas por la precisión, así que la BP solo necesita castigar las **demasiado cortas**. La fórmula:

$$
BP =
\begin{cases}
1 & \text{si } c > r \\[4pt]
e^{\,1 - r/c} & \text{si } c \le r
\end{cases}
$$

donde:

- $c$ es la **longitud total del corpus de candidatos** (la suma de las longitudes de todas las oraciones traducidas).
- $r$ es la **longitud efectiva de referencia** del corpus. Se calcula sumando, para cada oración candidata, la *best match length*: la longitud de la referencia **más cercana** a esa candidata. (Si una candidata mide 12 y hay referencias de 12, 15 y 17, la *best match length* es 12 y la BP de esa oración sería 1.)

Cuando el candidato es igual o más largo que la referencia ($c > r$), no hay penalización: $BP = 1$. Cuando es más corto ($c \le r$), la penalización es un **exponencial decreciente en $r/c$**: cuanto más corto sea el candidato relativo a la referencia, más se aleja $r/c$ de 1 y más fuerte es el castigo. Por ejemplo, si el candidato mide la mitad de la referencia ($c = r/2$, es decir $r/c = 2$), entonces $BP = e^{1-2} = e^{-1} \approx 0.368$.

### 5.4. Un detalle importante: BP a nivel de corpus

El paper insiste (sección 2.2.2) en que la BP se computa **sobre el corpus entero**, no oración por oración. Si se calculara por oración y se promediara, las desviaciones de longitud en oraciones cortas se castigarían **muy duramente**. Computarla sobre todo el corpus deja "algo de libertad" a nivel de oración: lo que importa es que la longitud total del candidato no se desvíe sistemáticamente de la longitud total de referencia.

---

## 6. La fórmula final de BLEU

Reuniendo las piezas —media geométrica de precisiones modificadas, multiplicada por la penalización por brevedad— se obtiene la fórmula que aparece en la sección 2.3 del paper:

$$
\text{BLEU} = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

En la línea base del paper se usa $N = 4$ y **pesos uniformes** $w_n = 1/N = 1/4$. La única normalización de texto que aplican antes de computar las precisiones es el *case folding* (pasar todo a minúsculas).

El paper también da la versión en el **dominio logarítmico**, que revela mejor el comportamiento de ranking:

$$
\log \text{BLEU} = \min\!\left(1 - \frac{r}{c},\ 0\right) + \sum_{n=1}^{N} w_n \log p_n
$$

Aquí el término $\min(1 - r/c, 0)$ es exactamente el logaritmo de la BP: vale 0 cuando $c > r$ (sin penalización) y $1 - r/c < 0$ cuando $c \le r$.

### 6.1. Ejemplo completo trabajado

Construyamos un BLEU-4 de extremo a extremo con números concretos. Supongamos que tras procesar un corpus pequeño obtuvimos:

$$
p_1 = 0.90,\quad p_2 = 0.60,\quad p_3 = 0.40,\quad p_4 = 0.25
$$

y que el corpus candidato mide $c = 950$ palabras contra una longitud efectiva de referencia $r = 1000$.

**Paso 1 — media geométrica de las precisiones.** Con $w_n = 1/4$:

$$
\sum_{n=1}^{4} w_n \log p_n = \tfrac14(\ln 0.90 + \ln 0.60 + \ln 0.40 + \ln 0.25)
$$

$$
= \tfrac14(-0.1054 - 0.5108 - 0.9163 - 1.3863) = \tfrac14(-2.9188) = -0.7297
$$

$$
\exp(-0.7297) = 0.4821
$$

(Equivalentemente, $\sqrt[4]{0.90 \cdot 0.60 \cdot 0.40 \cdot 0.25} = \sqrt[4]{0.054} \approx 0.482$.)

**Paso 2 — penalización por brevedad.** Como $c = 950 \le r = 1000$:

$$
BP = e^{\,1 - r/c} = e^{\,1 - 1000/950} = e^{\,1 - 1.0526} = e^{-0.0526} = 0.9488
$$

**Paso 3 — BLEU final.**

$$
\text{BLEU} = BP \cdot 0.4821 = 0.9488 \cdot 0.4821 = 0.4574
$$

Es decir, **BLEU ≈ 0.457** (a veces reportado como 45.7 en escala 0–100). Nota cómo la brevedad relativamente leve (5% más corto) solo descuenta un ~5%, mientras que el grueso del valor lo determina la media geométrica de precisiones, dominada por las precisiones bajas de los n-gramas largos.

---

## 7. BLEU a nivel de corpus vs. a nivel de oración

Este es uno de los puntos más sutiles y de los más mal entendidos en la práctica. El paper es explícito (secciones 2.1.1 y 2.2.2): aunque la **unidad básica de evaluación** es la oración, BLEU se computa **sobre todo el corpus de prueba acumulando numeradores y denominadores**, no promediando BLEU por oración.

El procedimiento es:

1. Para cada $n \in \{1,2,3,4\}$, recorrer todas las oraciones candidatas, contar coincidencias recortadas (numerador) y total de n-gramas candidatos (denominador), y **sumarlos a lo largo de todo el corpus** antes de dividir. Esto da los $p_n$ globales.
2. Acumular $c$ (longitud total de candidatos) y $r$ (suma de *best match lengths*) sobre todo el corpus para una única BP.
3. Combinar en un único número BLEU para el corpus.

¿Por qué importa? Dos implicaciones:

- **Robustez estadística.** BLEU "solo necesita coincidir con el juicio humano cuando se promedia sobre un corpus de prueba; las puntuaciones de oraciones individuales suelen variar mucho respecto del juicio humano" (nota al pie 4 del paper). Una oración que casualmente coincide con una frase fluida como "East Asian economy" inflaría su precisión de n-gramas largos, pero al promediar sobre todo el corpus, y al tratar a todos los sistemas igual con múltiples traductores de distintos estilos, ese efecto se cancela en las comparaciones entre sistemas. El paper lo resume: **la cantidad lleva a la calidad** —promediar errores individuales sobre un corpus es más confiable que adivinar el juicio humano de cada oración.
- **No se puede promediar BLEU por oración.** Calcular BLEU oración por oración y promediar da un número **distinto** (y peor) que el BLEU de corpus, sobre todo por dos razones: las oraciones con algún $p_n = 0$ colapsan a BLEU = 0 (la media geométrica es implacable) y la BP por oración castiga las cortas demasiado duro. Por eso el "sentence-level BLEU" requiere *smoothing* y es notoriamente ruidoso.

---

## 8. Experimentos de validación: ¿correlaciona con el juicio humano?

El paper dedica las secciones 3, 4 y 5 a demostrar lo único que importa de una métrica automática: **que prediga lo que dirían los humanos**.

### 8.1. El setup experimental

- **5 "sistemas":** tres sistemas comerciales de MT (S1, S2, S3) y dos traductores humanos (H1, H2). H1 era una persona **sin proficiencia nativa** ni en el idioma fuente (chino) ni en el destino (inglés); H2 era hablante nativo de inglés.
- **Corpus:** ~500 oraciones (40 noticias generales), traducción chino→inglés, con hasta 4 referencias.
- **Jueces humanos:** dos grupos de 10 personas cada uno. El grupo **monolingüe** (10 nativos de inglés) juzgaba solo legibilidad y fluidez; el grupo **bilingüe** (10 nativos de chino residentes en EE. UU.) podía juzgar también adecuación. Cada juez puntuó de 1 (muy malo) a 5 (muy bueno).

### 8.2. Puntuaciones BLEU de los 5 sistemas (Tabla 1)

Sobre las 500 oraciones, contra dos referencias:

| Sistema | S1 | S2 | S3 | H1 | H2 |
|---------|------|------|------|------|------|
| BLEU | 0.0527 | 0.0829 | 0.0930 | 0.1934 | 0.2571 |

El orden BLEU (S1 < S2 < S3 < H1 < H2) coincide con el orden de calidad. Nótese también que **incluso los traductores humanos no obtienen 1.0** —de hecho H2, el mejor, saca apenas 0.2571. BLEU rara vez se acerca a 1 a menos que la salida sea idéntica a una referencia, y como hay muchas traducciones válidas, ni un humano lo logra. (El paper también señala que un traductor humano puntuó 0.3468 contra cuatro referencias pero solo 0.2571 contra dos: **a más referencias, mayor BLEU**, por lo que comparar puntuaciones BLEU con distinto número de referencias es engañoso.)

### 8.3. Significancia estadística (Tabla 2)

¿Es confiable la diferencia entre sistemas tan parecidos como S2 y S3? Dividieron el corpus en 20 bloques de 25 oraciones, computaron BLEU por bloque, y corrieron t-tests pareados entre sistemas adyacentes. Un t-estadístico de 1.7 o más es significativo al 95%. Los resultados:

| | S1 | S2 | S3 | H1 | H2 |
|---|------|------|------|------|------|
| Media | 0.051 | 0.081 | 0.090 | 0.192 | 0.256 |
| Desv. est. | 0.017 | 0.025 | 0.020 | 0.030 | 0.039 |
| t | — | 6 | 3.4 | 24 | 11 |

Todas las diferencias son estadísticamente muy significativas (t ≥ 3.4 ≫ 1.7), incluido el delicado par S2 vs. S3.

### 8.4. La correlación con el juicio humano (sección 5)

El resultado estrella: una regresión lineal de las puntuaciones del grupo monolingüe en función de BLEU sobre los 5 sistemas da un **coeficiente de correlación de 0.99**. Para el grupo bilingüe, el coeficiente es **0.96**. BLEU **rastrea el juicio humano muy de cerca**, y —lo más impresionante— distingue correctamente entre S2 y S3 a pesar de su cercanía. El paper concluye que BLEU acelerará el ciclo de I+D en MT al permitir a los investigadores converger rápidamente hacia ideas de modelado efectivas, sustentado en correlaciones validadas para traducción al inglés desde cuatro idiomas distintos (árabe, chino, francés, español; tres familias lingüísticas), según su trabajo compañero (Papineni et al., 2002).

---

## 9. Limitaciones: lo que BLEU no ve

A pesar de su éxito, BLEU mide **similitud superficial de cadenas**, y ese es su techo. Las limitaciones —algunas reconocidas por el propio paper, otras evidenciadas por dos décadas de uso— son:

- **No captura significado ni semántica.** BLEU solo cuenta coincidencias de n-gramas. No tiene noción de sinónimos, de roles semánticos ni de si la traducción dice lo mismo que el original. Es literalmente la frase de la slide 27: "no evalúa la calidad de la traducción", solo la **similitud de cadenas**.
- **Penaliza paráfrasis válidas.** Una traducción perfectamente correcta que use sinónimos no presentes en las referencias (decir *automobile* donde la referencia dice *car*) recibe precisión baja injustamente. Cuantas más referencias, menos grave; pero las referencias siempre son finitas.
- **Sensible a la tokenización.** Como BLEU opera sobre tokens, decisiones de tokenización (cómo separar puntuación, contracciones, mayúsculas) cambian la puntuación. Esto hizo que durante años las cifras BLEU **no fueran comparables entre papers**, hasta que SacreBLEU (Post, 2018) estandarizó la tokenización.
- **Problemática a nivel de oración individual.** Como vimos en la sección 7, BLEU está diseñado para corpus. A nivel de oración es ruidoso y colapsa a cero ante cualquier $p_n = 0$.
- **No mide orden global ni coherencia de largo alcance.** Captura fluidez local vía n-gramas hasta $N=4$, pero no estructura del discurso ni reordenamientos legítimos más allá de esa ventana.

Estas grietas son la razón directa por la que surgieron métricas posteriores: **METEOR** (Banerjee y Lavie, 2005) incorpora *stemming*, sinónimos vía WordNet y recall explícito; **BERTScore** (Zhang et al., 2020) reemplaza el conteo de n-gramas por **similitud coseno de embeddings contextuales**, capturando paráfrasis y semántica que BLEU es ciego a ver; **BLEURT** (Sellam et al., 2020) entrena un modelo para predecir directamente el juicio humano. Todas atacan el mismo punto: BLEU mide superficie, no significado.

---

## 10. BLEU en Image Captioning (la conexión con la clase 23)

Aunque BLEU nació para MT, la sección final del paper ya anticipaba su generalización: "dado que MT y *summarization* pueden verse ambas como generación de lenguaje natural en un contexto textual, creemos que BLEU podría adaptarse para evaluar *summarization* u otras tareas de NLG similares." Esa profecía se cumplió de sobra: BLEU es hoy estándar en **Image Captioning**.

### 10.1. Cómo se adapta

La adaptación es casi mecánica. En *captioning* no hay una oración fuente en otro idioma, pero sí hay el ingrediente esencial de BLEU: **múltiples referencias humanas**. En el benchmark **MS COCO Captions**, cada imagen tiene **5 captions humanos**. Esos 5 captions juegan exactamente el papel de las 5 referencias de traducción: el caption generado por el modelo es el "candidato", y se computa la precisión de n-gramas recortada contra los 5 captions, más la BP por brevedad.

Los papers de *captioning* reportan típicamente **BLEU-1, BLEU-2, BLEU-3 y BLEU-4** —es decir, BLEU usando $N=1,2,3,4$ respectivamente. BLEU-1 mide básicamente cobertura de vocabulario (¿menciona las palabras correctas?), mientras que BLEU-4 exige coincidencias de frases de 4 palabras y es mucho más exigente con la fluidez. Es habitual ver tablas con las cuatro cifras a la vez.

### 10.2. Limitaciones específicas en captioning

En *captioning* las debilidades de BLEU son aún más agudas que en MT:

- **Premia repetir palabras frecuentes.** Como los captions de COCO comparten mucho vocabulario genérico (*a man*, *a person*, *standing*, *on a*), un modelo puede subir BLEU produciendo captions genéricos y seguros sin describir lo distintivo de la imagen.
- **No mide relevancia visual.** BLEU compara cadenas; no tiene acceso a la imagen. Un caption gramaticalmente fluido que describe la imagen equivocada puede sacar un BLEU decente si comparte n-gramas con las referencias.
- **Mal correlacionada a nivel de imagen individual.** Igual que a nivel de oración en MT, BLEU por imagen es ruidoso.

Por eso la comunidad de *captioning* desarrolló métricas **específicas**: **CIDEr** (Vedantam et al., 2015) pondera los n-gramas por **TF-IDF** sobre el conjunto de referencias, premiando los n-gramas informativos y distintivos por encima de los genéricos —exactamente el agujero de BLEU; **SPICE** (Anderson et al., 2016) compara **grafos de escena** (objetos, atributos y relaciones), acercándose a la semántica visual; y **METEOR** se usa también en *captioning* por su manejo de sinónimos. En la práctica, los papers de *captioning* reportan el cuarteto BLEU-1..4 junto con METEOR, CIDEr y SPICE, porque ninguna métrica sola captura todo.

---

## 11. Impacto y legado

BLEU es, sin exagerar, una de las métricas más influyentes de la historia del NLP. Durante **más de 20 años** fue la métrica de facto para reportar resultados en traducción automática y en *image captioning*, y su número fue la moneda de cambio de miles de papers. Su éxito se explica por la combinación de tres propiedades: es **barata** (costo marginal casi nulo), **rápida** (segundos) y **suficientemente correlacionada** con el juicio humano para guiar la iteración de investigación.

Su legado va más allá de su uso directo. BLEU **definió el paradigma** de evaluación automática de NLG por comparación con referencias humanas: la idea de precisión de n-gramas recortada, de combinar escalas con media geométrica y de penalizar la brevedad reaparece, transformada, en casi toda métrica posterior. ROUGE (ya en el site) es esencialmente "BLEU pero recall-oriented" para *summarization*. METEOR, NIST, CIDEr, chrF y muchas otras son variaciones sobre el tema que BLEU inauguró.

Incluso en la era de los **LLM**, BLEU sigue vivo: aparece en *benchmarks* de traducción, se usa como señal barata en *ablations*, y es el punto de comparación histórico obligatorio. Aunque para evaluación de calidad fina hoy se prefieren métricas neuronales (COMET, BLEURT, BERTScore) o *LLM-as-a-judge*, BLEU permanece como la línea base universal: si tu sistema no supera BLEU, no lo va a superar nadie.

---

## 12. Conexión con la clase 23

En la clase 23 sobre **Image Captioning**, BLEU aparece en el **slide 27** como la métrica para evaluar las descripciones generadas. La descripción de la slide —"puntúa de 0 a 1, se centra en la similitud de cadenas, no evalúa la calidad de la traducción"— es una síntesis exacta de este paper:

- **"de 0 a 1":** la sección 3 del paper lo dice textualmente, *"The BLEU metric ranges from 0 to 1"*, y aclara que pocas traducciones alcanzan 1 a menos que sean idénticas a una referencia.
- **"similitud de cadenas":** BLEU cuenta coincidencias de n-gramas; es similitud superficial, no semántica.
- **"no evalúa la calidad de la traducción":** BLEU no entiende el significado; es un *understudy* estadístico que **correlaciona** con la calidad sin medirla directamente.

BLEU complementa el otro hilo de la clase: la **generación** de captions. Los slides 24–26 cubren las estrategias de decodificación (*greedy search* vs. *beam search*), que deciden **cómo** el modelo produce el caption palabra por palabra. BLEU es lo que viene después: una vez generado el caption con greedy o beam, BLEU **mide** qué tan bueno es contra los captions humanos de referencia. Generación (slides 24–26) y evaluación (slide 27) son las dos caras del problema de NLG en *captioning*.

---

## 13. Notas y enlaces

- **Relación con ROUGE** (ver `rouge-metric.md` en el site): ROUGE es el espejo recall-oriented de BLEU. BLEU normaliza por el **candidato** (precision: ¿lo que generé es correcto?), ROUGE normaliza por la **referencia** (recall: ¿cubrí el contenido?). BLEU domina en traducción y *captioning*; ROUGE domina en *summarization*. Ambas comparten la misma raíz: conteo de n-gramas contra referencias humanas, validado por correlación con jueces.
- **METEOR** (Banerjee y Lavie, 2005): añade *stemming*, sinónimos (WordNet) y recall explícito; corrige la ceguera de BLEU a paráfrasis. Muy usada también en *captioning*.
- **CIDEr** (Vedantam et al., 2015): específica de *captioning*; pondera n-gramas por TF-IDF para premiar lo distintivo sobre lo genérico —el agujero exacto de BLEU en COCO.
- **SPICE** (Anderson et al., 2016): compara grafos de escena (objetos/atributos/relaciones); se acerca a la semántica visual que BLEU no ve.
- **BERTScore / BLEURT / COMET**: métricas neuronales que reemplazan el conteo de n-gramas por similitud de embeddings o predicción aprendida del juicio humano; capturan significado y paráfrasis. Son los sucesores de la era LLM.
- **SacreBLEU** (Post, 2018): no es una métrica nueva sino una **implementación estandarizada** de BLEU con tokenización fija, que resolvió el problema de comparabilidad entre papers. Si reportas BLEU hoy, repórtalo con SacreBLEU.
- **Detalle de implementación heredado del paper:** $N=4$, pesos uniformes $w_n = 1/4$, *case folding* como única normalización, y cómputo a nivel de **corpus** (no de oración). El *smoothing* para el caso $p_n = 0$ (Chen y Cherry, 2014) es un añadido posterior, ausente del paper original.

---

**Referencia primaria:** Papineni, K., Roukos, S., Ward, T., y Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL 2002, pp. 311–318. ACL Anthology P02-1040. <https://aclanthology.org/P02-1040/>
