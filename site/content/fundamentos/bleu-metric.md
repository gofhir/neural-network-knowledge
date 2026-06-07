---
title: "BLEU Metric"
weight: 94
math: true
---

**BLEU** (Bilingual Evaluation Understudy) es la métrica **precision-oriented** más usada para evaluar **traducción automática** (Machine Translation, MT) y, por extensión, **image captioning** y otras tareas de generación de lenguaje natural que comparan una salida contra referencias humanas. Fue propuesta por Kishore Papineni, Salim Roukos, Todd Ward y Wei-Jing Zhu (IBM T. J. Watson Research Center) en el paper [BLEU: a Method for Automatic Evaluation of Machine Translation (ACL 2002)](https://aclanthology.org/P02-1040/).

La motivación era práctica y casi idéntica a la que años después daría origen a ROUGE: la evaluación humana de traducciones era lenta, cara y no reutilizable. Los autores propusieron un **understudy** —un suplente estadístico— que correlaciona con el juicio humano y se computa en segundos. Más de veinte años después, BLEU sigue siendo el reporte obligatorio en cualquier paper de traducción y aparece junto a CIDEr, METEOR y SPICE en *captioning*, aunque sus limitaciones son bien conocidas y motivaron una larga lista de sucesores (METEOR, BERTScore, BLEURT, COMET).

Este fundamento cubre por qué BLEU es **precision-oriented** (en contraste con el recall de ROUGE), la **modified n-gram precision con clipping**, la **media geométrica** de las precisiones, la **brevity penalty**, la fórmula final con ejemplos numéricos trabajados paso a paso, la distinción **corpus-level vs sentence-level**, el uso en **image captioning** y las limitaciones que motivaron las métricas semánticas. Es fundamento de la [Clase 23](/clases/clase-23) y complementa el [fundamento ROUGE](/fundamentos/rouge-metric).

---

## 1. ¿Por qué precision-oriented?

El nombre delata la mitad de la filosofía: **Bilingual Evaluation Understudy**. La otra mitad está en cómo se construye la fórmula. BLEU pregunta, esencialmente: *de todo lo que generó el modelo, ¿qué fracción es correcta?* Eso es **precisión**: el denominador cuenta los n-gramas del **candidato**, no del reference.

La intuición es propia de la traducción. Si el modelo traduce una oración fuente y produce diez palabras, lo que importa es que esas diez palabras sean **correctas** —que aparezcan en alguna traducción humana de referencia—. No nos preocupa tanto que el candidato "cubra" todo el contenido posible de las referencias, porque en MT hay una oración fuente fija y el espacio de traducciones válidas es acotado: una buena traducción dice lo que hay que decir y nada más.

{{< concept-alert type="clave" >}}
**BLEU = precision sobre n-gramas del candidato**. El denominador en la fórmula básica es la cuenta de n-gramas en el candidato (lo que generó el modelo), no en las referencias humanas. Esto es exactamente lo opuesto a ROUGE, que normaliza por el reference (recall).
{{< /concept-alert >}}

Compáralo con ROUGE (Lin 2004), que es recall-oriented y reina en *summarization*: en un resumen lo importante es **cubrir el contenido del reference** —si dejas afuera información clave, fallaste, aunque lo que escribiste sea correcto—. En MT la balanza se invierte: lo importante es que **lo que generaste sea correcto**, porque la fuente ya fija qué hay que decir. Ambas métricas comparten la misma raíz —conteo de n-gramas recortados contra referencias humanas, validado por correlación con jueces— pero apuntan a extremos opuestos del trade-off precision/recall.

Esta elección no es gratuita: la precisión sola tiene un agujero gigantesco por el lado de la **brevedad** (un candidato cortísimo puede tener precisión perfecta), y BLEU lo tapa con la *brevity penalty* en lugar de con recall, por razones que veremos en la sección 5.

---

## 2. El problema formal

Sea $C$ el **candidato** (la traducción producida por el modelo) y $R = \{R_1, \ldots, R_k\}$ el conjunto de **referencias** escritas por traductores humanos profesionales (típicamente $k \in \{1, 4\}$; el corpus original del paper usaba hasta 4 referencias por oración fuente).

Queremos una función:

{{< math-formula title="Score de evaluación" >}}
\text{BLEU}(C, R) \in [0, 1]
{{< /math-formula >}}

que sea **alta** cuando $C$ se parece a alguna combinación de los $R_i$ y **baja** cuando no, y cuya correlación con el juicio humano sea lo más alta posible. El paper lo formula como un *viewpoint*: "cuanto más cerca esté una traducción automática de una traducción humana profesional, mejor es". Operacionalizar esa intuición requiere dos ingredientes:

1. Una **métrica numérica de cercanía de traducción** (*translation closeness*).
2. Un **corpus de referencias humanas** de buena calidad.

El ejemplo canónico del paper (Example 1) ilustra el problema. La oración fuente es china; hay dos candidatos:

- **Candidato 1 (bueno):** *It is a guide to action which ensures that the military always obeys the commands of the party.*
- **Candidato 2 (malo):** *It is to insure the troops forever hearing the activity guidebook that party direct.*

y tres referencias humanas:

- **Referencia 1:** *It is a guide to action that ensures that the military will forever heed Party commands.*
- **Referencia 2:** *It is the guiding principle which guarantees the military forces always being under the command of the Party.*
- **Referencia 3:** *It is the practical guide for the army always to heed the directions of the party.*

El Candidato 1 comparte muchas frases con las tres referencias; el Candidato 2 casi ninguna. La tarea de BLEU es **comparar los n-gramas del candidato con los de las referencias y contar coincidencias**, independientes de la posición. Cuantas más coincidencias, mejor el candidato.

A diferencia de ROUGE, que es una **familia** de métricas (N, L, W, S, SU), BLEU es esencialmente **una sola** métrica con un parámetro $N$ (el largo máximo de n-grama, casi siempre 4). Toda la riqueza de BLEU vive en cómo combina las precisiones de $p_1$ a $p_4$ y en cómo penaliza la brevedad.

---

## 3. Modified n-gram precision con clipping

### 3.1. Por qué la precisión ingenua se rompe

La piedra angular de BLEU es la **precisión de n-gramas**. En su versión ingenua, la precisión de unigramas es: *cuenta cuántas palabras del candidato aparecen en alguna referencia, y divide por el total de palabras del candidato.*

El problema es que los sistemas de MT pueden **sobregenerar** palabras "razonables", produciendo traducciones absurdas pero de altísima precisión. El ejemplo (Example 2) es el clásico de toda clase de NLP:

- **Candidato:** *the the the the the the the.* (7 ocurrencias de *the*)
- **Referencia 1:** *The cat is on the mat.*
- **Referencia 2:** *There is a cat on the mat.*

La palabra *the* aparece en las referencias, así que las 7 ocurrencias del candidato "cuentan". La precisión de unigramas ingenua sería:

$$
p_1^{\text{ingenua}} = \frac{7}{7} = 1.0
$$

Una puntuación **perfecta** para una traducción que no significa nada. La métrica está claramente rota.

### 3.2. El clipping: la modificación clave

La solución del paper es el **conteo recortado** (*clipping*). La intuición que formalizan: una palabra de la referencia se considera **agotada** una vez que ya fue emparejada por una palabra del candidato. Operativamente, para cada n-grama del candidato:

1. Calcula su conteo en el candidato, $\text{Count}$.
2. Calcula el **máximo número de veces** que ese n-grama aparece en **una sola** referencia. Llámalo $\text{Max\_Ref\_Count}$.
3. Recorta el conteo del candidato a ese máximo:

$$
\text{Count}_{\text{clip}} = \min(\text{Count},\ \text{Max\_Ref\_Count})
$$

4. Suma los conteos recortados de todos los n-gramas únicos del candidato y divide por el **total de n-gramas del candidato** (sin recortar).

Para el ejemplo de *the the the…*: en la Referencia 1 aparece *the* dos veces (*The* y *the*); en la Referencia 2 una vez. El máximo en una sola referencia es 2. Entonces:

$$
\text{Count}_{\text{clip}}(\text{the}) = \min(7, 2) = 2
$$

y la **precisión de unigramas modificada** es:

$$
p_1 = \frac{2}{7} \approx 0.286
$$

frente al $1.0$ de la versión ingenua. El absurdo queda penalizado. Este clipping es exactamente el mismo mecanismo que ROUGE-N usa en su $\text{Count}_{\text{match}}$; lo que cambia es el denominador (candidato en BLEU, reference en ROUGE).

### 3.3. La fórmula general

Para un n-grama de cualquier longitud, y sobre un **corpus completo** de candidatos, la precisión de n-gramas modificada es:

{{< math-formula title="Modified n-gram precision" >}}
p_n = \frac{\displaystyle\sum_{\mathcal{C}\,\in\,\{\text{Candidatos}\}} \ \sum_{\text{n-gram}\,\in\,\mathcal{C}} \text{Count}_{\text{clip}}(\text{n-gram})}{\displaystyle\sum_{\mathcal{C}'\,\in\,\{\text{Candidatos}\}} \ \sum_{\text{n-gram}'\,\in\,\mathcal{C}'} \text{Count}(\text{n-gram}')}
{{< /math-formula >}}

donde el numerador acumula conteos **recortados** sobre todos los n-gramas de todas las oraciones candidatas del corpus, y el denominador acumula los conteos **sin recortar** (el total de n-gramas candidatos).

### 3.4. Ejemplo numérico trabajado del paper

El paper reporta las precisiones modificadas de los candidatos del Example 1:

| Candidato | $p_1$ (unigramas) | $p_2$ (bigramas) |
|---|---|---|
| Candidato 1 (bueno) | $17/18 \approx 0.944$ | $10/17 \approx 0.588$ |
| Candidato 2 (malo) | $8/14 \approx 0.571$ | $1/13 \approx 0.077$ |

El contraste es nítido: el buen candidato empareja casi todos sus unigramas y más de la mitad de sus bigramas; el malo cae a la mitad en unigramas y se desploma en bigramas. El paper observa que esto captura **dos aspectos** de la traducción simultáneamente: los unigramas miden **adecuación** (¿están las palabras correctas?), mientras que los n-gramas más largos miden **fluidez** (¿están en un orden gramatical razonable?). Es la misma distinción que en ROUGE separa ROUGE-1 (informatividad) de ROUGE-2 (fluencia local).

{{< concept-alert type="ojo" >}}
La precisión modificada ya penaliza candidatos demasiado **largos**: cada palabra espuria que no aparece en ninguna referencia infla el denominador sin sumar al numerador, y el clipping castiga repetir una palabra más veces de las que aparece en las referencias. Pero **no penaliza candidatos demasiado cortos** —ese agujero lo tapa la brevity penalty, no la precisión—.
{{< /concept-alert >}}

---

## 4. Combinar precisiones: por qué media geométrica

Hasta aquí tenemos cuatro números separados $p_1, p_2, p_3, p_4$. ¿Cómo combinarlos en una sola cifra?

La clave es una observación empírica del paper: **la precisión modificada decae aproximadamente de forma exponencial con $n$**. La precisión de unigramas es mucho mayor que la de bigramas, que a su vez es mucho mayor que la de trigramas, etcétera. Tiene sentido: es fácil acertar palabras sueltas, difícil acertar secuencias largas idénticas.

Si usáramos un **promedio aritmético** $\frac{1}{4}(p_1 + p_2 + p_3 + p_4)$, las precisiones grandes de $n$ bajo dominarían y las diferencias en los n-gramas largos —justo donde vive la señal de fluidez— quedarían diluidas. Un esquema de promediado razonable debe **tener en cuenta este decaimiento exponencial**. La respuesta del paper es promediar **el logaritmo** de las precisiones con pesos uniformes, lo que equivale a la **media geométrica**:

{{< math-formula title="Media geométrica de las precisiones" >}}
\left(\prod_{n=1}^{N} p_n\right)^{1/N} = \exp\!\left(\frac{1}{N}\sum_{n=1}^{N} \log p_n\right)
{{< /math-formula >}}

El logaritmo convierte el decaimiento multiplicativo en algo aditivo y tratable, y la media geométrica da un peso relativo justo a cada escala. El paper reporta dos hallazgos:

1. La media geométrica **correlaciona algo mejor** con el juicio humano que la aritmética.
2. La media geométrica es **dura**: si **cualquier** $p_n$ se hace cero, todo el producto colapsa a cero. Los autores argumentan que en corpus de tamaño razonable con $N_{\max} \le 4$ eso es un evento extremadamente raro.

{{< concept-alert type="ojo" >}}
Esa dureza es exactamente la razón por la que el **sentence-level BLEU** (un candidato de una sola oración) es problemático: basta que no haya ni un solo 4-grama en común para que $p_4 = 0$ y el BLEU de esa oración colapse a 0, aunque la traducción sea decente. La solución moderna es el *smoothing* de Chen y Cherry (2014), ausente del paper original. Es el análogo al colapso de la media geométrica que ROUGE evita reportando F1 por variante en lugar de combinarlas.
{{< /concept-alert >}}

---

## 5. Brevity Penalty (BP): el agujero del recall

### 5.1. El problema: la precisión no castiga candidatos cortos

La precisión de n-gramas modificada falla en un extremo: **no castiga candidatos demasiado cortos**. El ejemplo (Example 3) lo muestra con las mismas tres referencias del Example 1:

- **Candidato:** *of the*

Como el candidato es minúsculo, sus precisiones se **inflan**: precisión de unigramas $= 2/2 = 1.0$ y precisión de bigramas $= 1/1 = 1.0$. Una traducción de dos palabras obtiene precisión perfecta. Es el problema simétrico al de *the the the*.

### 5.2. Por qué no se usa recall

La forma tradicional de tapar este agujero sería **emparejar precisión con recall** —exactamente lo que hace ROUGE—. Pero el paper argumenta que el recall **no funciona** cuando hay múltiples referencias. El ejemplo (Example 4):

- **Candidato 1:** *I always invariably perpetually do.*
- **Candidato 2:** *I always do.*
- **Referencia 1:** *I always do.* · **Referencia 2:** *I invariably do.* · **Referencia 3:** *I perpetually do.*

El Candidato 1 "recuerda" más palabras de las referencias (*always*, *invariably*, *perpetually*), pero es obviamente **peor** que el Candidato 2. Recordar **todas** las opciones sinónimas a la vez produce una traducción mala. Por lo tanto, un recall ingenuo sobre el conjunto de todas las palabras de referencia es inadecuado, y calcular recall sobre conceptos (alineando sinónimos) es demasiado complicado dado que las referencias varían en longitud y sintaxis. De ahí que BLEU evite el recall y use otra cosa.

### 5.3. La solución: penalización por brevedad

En lugar de recall, BLEU introduce un **factor multiplicativo de penalización por brevedad** (*brevity penalty*, BP). Como las traducciones más **largas** que la referencia ya están castigadas por la precisión, la BP solo necesita castigar las **demasiado cortas**:

{{< math-formula title="Brevity Penalty" >}}
BP =
\begin{cases}
1 & \text{si } c > r \\[4pt]
e^{\,1 - r/c} & \text{si } c \le r
\end{cases}
{{< /math-formula >}}

donde:

- $c$ es la **longitud total del corpus de candidatos** (la suma de las longitudes de todas las oraciones traducidas).
- $r$ es la **longitud efectiva de referencia** del corpus. Se calcula sumando, para cada oración candidata, la *best match length*: la longitud de la referencia **más cercana** a esa candidata. (Si una candidata mide 12 palabras y hay referencias de 12, 15 y 17, la *best match length* es 12.)

Cuando el candidato es igual o más largo que la referencia ($c > r$), no hay penalización: $BP = 1$. Cuando es más corto ($c \le r$), la penalización es un **exponencial decreciente en $r/c$**: cuanto más corto sea el candidato relativo a la referencia, más se aleja $r/c$ de 1 y más fuerte el castigo. Por ejemplo, si el candidato mide la mitad de la referencia ($c = r/2$, es decir $r/c = 2$):

$$
BP = e^{\,1 - 2} = e^{-1} \approx 0.368
$$

un descuento del 63% sobre la media geométrica de precisiones.

### 5.4. Un detalle importante: BP a nivel de corpus

El paper insiste en que la BP se computa **sobre el corpus entero**, no oración por oración. Si se calculara por oración y se promediara, las desviaciones de longitud en oraciones cortas se castigarían **muy duramente**. Computarla sobre todo el corpus deja "algo de libertad" a nivel de oración: lo que importa es que la longitud total del candidato no se desvíe sistemáticamente de la longitud total de referencia.

---

## 6. La fórmula final

Reuniendo las piezas —media geométrica de precisiones modificadas, multiplicada por la penalización por brevedad— se obtiene la fórmula de la sección 2.3 del paper:

{{< math-formula title="BLEU" >}}
\text{BLEU} = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
{{< /math-formula >}}

En la línea base del paper se usa $N = 4$ y **pesos uniformes** $w_n = 1/N = 1/4$. La única normalización de texto que se aplica antes de computar las precisiones es el *case folding* (pasar todo a minúsculas).

El paper también da la versión en el **dominio logarítmico**, que revela mejor el comportamiento de ranking:

$$
\log \text{BLEU} = \min\!\left(1 - \frac{r}{c},\ 0\right) + \sum_{n=1}^{N} w_n \log p_n
$$

Aquí el término $\min(1 - r/c, 0)$ es exactamente el logaritmo de la BP: vale $0$ cuando $c > r$ (sin penalización) y $1 - r/c < 0$ cuando $c \le r$.

### 6.1. Ejemplo completo trabajado paso a paso

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

Es decir, **BLEU ≈ 0.457** (a veces reportado como 45.7 en escala 0–100). Nota cómo la brevedad relativamente leve (5% más corto) solo descuenta un ~5%, mientras que el grueso del valor lo determina la media geométrica de precisiones, dominada por las precisiones bajas de los n-gramas largos. Si en cambio $p_4$ hubiera sido $0$, todo el BLEU colapsaría a $0$ por la dureza de la media geométrica —un recordatorio de por qué a nivel de corpus esto rara vez pasa, pero a nivel de oración es un riesgo real.

---

## 7. Corpus-level vs sentence-level

Este es uno de los puntos más sutiles y peor entendidos en la práctica. El paper es explícito: aunque la **unidad básica de evaluación** es la oración, BLEU se computa **sobre todo el corpus de prueba acumulando numeradores y denominadores**, no promediando BLEU por oración.

El procedimiento es:

1. Para cada $n \in \{1,2,3,4\}$, recorrer todas las oraciones candidatas, contar coincidencias recortadas (numerador) y total de n-gramas candidatos (denominador), y **sumarlos a lo largo de todo el corpus** antes de dividir. Esto da los $p_n$ globales.
2. Acumular $c$ (longitud total de candidatos) y $r$ (suma de *best match lengths*) sobre todo el corpus para una única BP.
3. Combinar en un único número BLEU para el corpus.

¿Por qué importa? Dos implicaciones:

- **Robustez estadística.** El paper observa que BLEU "solo necesita coincidir con el juicio humano cuando se promedia sobre un corpus de prueba; las puntuaciones de oraciones individuales suelen variar mucho respecto del juicio humano". Una oración que casualmente empareja una frase fluida inflaría su precisión de n-gramas largos, pero al promediar sobre todo el corpus ese efecto se cancela. El lema del paper: **la cantidad lleva a la calidad**.
- **No se puede promediar BLEU por oración.** Calcular BLEU oración por oración y promediar da un número **distinto** (y peor) que el BLEU de corpus, por dos razones: las oraciones con algún $p_n = 0$ colapsan a BLEU $= 0$ (la media geométrica es implacable) y la BP por oración castiga las cortas demasiado duro.

{{< concept-alert type="clave" >}}
Por eso el **sentence-level BLEU** requiere *smoothing* (Chen y Cherry, 2014): se suma una pequeña cantidad a los conteos de n-gramas para evitar que $p_n = 0$ colapse todo. Sin smoothing, BLEU por oración es notoriamente ruidoso. Es el mismo patrón que en ROUGE: las métricas de overlap están diseñadas para agregarse sobre muchos ejemplos, no para confiar en un score aislado.
{{< /concept-alert >}}

Como referencia empírica del paper: sobre 500 oraciones chino→inglés con dos referencias, tres sistemas comerciales (S1, S2, S3) y dos humanos (H1, H2) obtuvieron BLEU de $0.0527$, $0.0829$, $0.0930$, $0.1934$ y $0.2571$ respectivamente —el mismo orden que asignaron los jueces humanos, con una correlación de Pearson de **0.99** para el grupo monolingüe. Nota que ni siquiera el mejor humano alcanza $1.0$: como hay muchas traducciones válidas, ningún candidato empareja perfectamente las referencias.

---

## 8. BLEU en Image Captioning

Aunque BLEU nació para MT, la sección final del paper ya anticipaba su generalización: "dado que MT y *summarization* pueden verse ambas como generación de lenguaje natural, creemos que BLEU podría adaptarse para evaluar *summarization* u otras tareas de NLG similares". Esa profecía se cumplió de sobra: BLEU es hoy estándar en **Image Captioning** (ver [Image Captioning](/fundamentos/image-captioning)).

### 8.1. Cómo se adapta

La adaptación es casi mecánica. En *captioning* no hay una oración fuente en otro idioma, pero sí está el ingrediente esencial de BLEU: **múltiples referencias humanas**. En el benchmark **MS COCO Captions**, cada imagen tiene **5 captions humanos**. Esos 5 captions juegan exactamente el papel de las 5 referencias de traducción: el caption generado por el modelo es el "candidato", y se computa la precisión de n-gramas recortada contra los 5 captions, más la BP por brevedad.

Los papers de *captioning* reportan típicamente el cuarteto **BLEU-1, BLEU-2, BLEU-3 y BLEU-4** —es decir, BLEU usando $N = 1, 2, 3, 4$ respectivamente—:

| Variante | $N$ | Qué exige | Qué mide |
|---|---|---|---|
| **BLEU-1** | 1 | Coincidencias de palabras sueltas | Cobertura de vocabulario |
| **BLEU-2** | 2 | Pares consecutivos | Fluencia local |
| **BLEU-3** | 3 | Frases de 3 palabras | Fluencia de frase |
| **BLEU-4** | 4 | Frases de 4 palabras | El más exigente; el reporte estrella |

BLEU-1 mide básicamente si el modelo menciona las palabras correctas; BLEU-4 exige coincidencias de frases de 4 palabras y es mucho más exigente con la fluidez. Es habitual ver tablas con las cuatro cifras a la vez, junto con METEOR, CIDEr y SPICE.

### 8.2. Limitaciones específicas en captioning

En *captioning*, las debilidades de BLEU son aún más agudas que en MT:

- **Premia repetir palabras frecuentes.** Como los captions de COCO comparten mucho vocabulario genérico (*a man*, *a person*, *standing*, *on a*), un modelo puede subir BLEU produciendo captions genéricos y seguros sin describir lo distintivo de la imagen.
- **No mide relevancia visual.** BLEU compara cadenas; no tiene acceso a la imagen. Un caption gramaticalmente fluido que describe la imagen equivocada puede sacar un BLEU decente si comparte n-gramas con las referencias.
- **Mal correlacionada a nivel de imagen individual.** Igual que a nivel de oración en MT, BLEU por imagen es ruidoso.

Por eso la comunidad de *captioning* desarrolló métricas específicas: **CIDEr** (Vedantam et al., 2015) pondera los n-gramas por **TF-IDF** sobre el conjunto de referencias, premiando los n-gramas informativos por encima de los genéricos —exactamente el agujero de BLEU—; **SPICE** (Anderson et al., 2016) compara **grafos de escena** (objetos, atributos y relaciones), acercándose a la semántica visual; y **METEOR** se usa también por su manejo de sinónimos. En la práctica los papers reportan el cuarteto BLEU-1..4 junto a METEOR, CIDEr y SPICE, porque ninguna métrica sola captura todo.

---

## 9. Limitaciones y sucesores

A pesar de su éxito, BLEU mide **similitud superficial de cadenas**, y ese es su techo. La slide 27 de la clase 23 lo resume sin ambages: "puntúa de 0 a 1, se centra en la similitud de cadenas, **no evalúa la calidad de la traducción**". Esa frase, que parece una crítica, es en realidad una cita casi literal de la filosofía del paper: BLEU no entiende nada de significado; solo mide cuánto se **parece superficialmente** un candidato a las referencias humanas.

| Limitación | Ejemplo | Por qué falla |
|---|---|---|
| **No captura significado** | "dice lo correcto con otras palabras" | Solo cuenta n-gramas; no hay semántica |
| **Penaliza paráfrasis válidas** | `automobile` donde la referencia dice `car` | Tokens distintos, no matchea |
| **Sinónimos** | `physician` vs `doctor` | Sin WordNet ni embeddings, no se reconocen |
| **Sensible a tokenización** | puntuación, contracciones, casing | Cambia la puntuación entre implementaciones |
| **Ruidoso a nivel de oración** | un solo $p_n = 0$ | La media geométrica colapsa a $0$ |
| **No mide orden global** | reordenamientos más allá de $N=4$ | La ventana de n-gramas es local |

La sensibilidad a la tokenización fue tan grave que durante años las cifras BLEU **no eran comparables entre papers**, hasta que **SacreBLEU** (Post, 2018) estandarizó la tokenización. SacreBLEU no es una métrica nueva sino una **implementación de referencia**: si reportas BLEU hoy, repórtalo con SacreBLEU.

{{< concept-alert type="clave" >}}
La grieta de fondo es la misma de ROUGE: **BLEU mide superficie, no significado**. Una traducción perfecta que use sinónimos no presentes en las referencias recibe precisión baja injustamente. Cuantas más referencias, menos grave; pero las referencias siempre son finitas.
{{< /concept-alert >}}

Estas grietas motivaron directamente a los sucesores:

| Métrica | Año | Mecanismo | Qué corrige de BLEU |
|---|---|---|---|
| **METEOR** (Banerjee-Lavie) | 2005 | Match exacto + stemming + sinónimos WordNet + recall | Paráfrasis, inflexión, recall explícito |
| **NIST** | 2002 | BLEU con n-gramas ponderados por información | Premia n-gramas raros e informativos |
| **CIDEr** (Vedantam) | 2015 | TF-IDF de n-gramas multi-referencia | Genericidad en captioning |
| **BERTScore** (Zhang) | 2020 | Cosine sim de embeddings BERT por token | Captura paráfrasis y semántica |
| **BLEURT** (Sellam) | 2020 | BERT fine-tuneado para predecir score humano | Aprende el juicio humano directamente |
| **COMET** (Rei) | 2020 | Modelo neuronal entrenado sobre juicios humanos de MT | SOTA en evaluación de traducción |

{{< concept-alert type="ojo" >}}
**No abandones BLEU.** Sigue vivo incluso en la era de los LLM: aparece en *benchmarks* de traducción, se usa como señal barata en *ablations*, y es el punto de comparación histórico obligatorio. La práctica moderna es **reportar BLEU + una métrica neuronal (COMET o BERTScore)**. Si tu sistema no supera el BLEU de la baseline, no lo va a superar nadie; pero un BLEU alto ya no basta para afirmar calidad.
{{< /concept-alert >}}

---

## 10. BLEU vs ROUGE: dos caras de la misma moneda

BLEU y ROUGE comparten ADN —conteo de n-gramas recortados contra referencias humanas, validado por correlación con jueces— pero se sitúan en extremos opuestos del trade-off precision/recall:

| Aspecto | BLEU | ROUGE |
|---|---|---|
| **Orientación** | Precision (¿lo generado es correcto?) | Recall (¿se cubrió el contenido?) |
| **Denominador** | n-gramas del **candidato** | n-gramas del **reference** |
| **Tarea dominante** | Traducción, image captioning | Summarization |
| **Combinación de escalas** | Media geométrica de $p_1..p_4$ | F1 por variante (N, L, S) |
| **Penalización de longitud** | Brevity penalty (castiga lo corto) | Implícita en el recall |
| **Colapso a 0** | Si algún $p_n = 0$ | Por variante, no colapsa global |
| **Año / autor** | 2002, Papineni et al. (IBM) | 2004, Lin (ISI/USC) |

La regla mnemotécnica: en MT importa que **lo que dijiste sea correcto** (precision → BLEU); en summarization importa que **cubras lo importante del reference** (recall → ROUGE). Por eso ROUGE es esencialmente "BLEU pero recall-oriented", y de hecho su nombre —**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation— es un homenaje explícito al *understudy* de BLEU.

La conexión con la [Clase 23](/clases/clase-23) cierra el círculo: las [estrategias de decodificación](/fundamentos/decoding-strategies) (greedy, beam search) deciden **cómo** el modelo genera el caption palabra por palabra; BLEU es lo que viene **después**, midiendo qué tan bueno es el caption generado contra las referencias humanas. Generación y evaluación son las dos caras del problema de NLG en *captioning*.

---

## Recursos relacionados

- [Clase 23](/clases/clase-23) — Image Captioning: generación (decoding) y evaluación (BLEU) de descripciones.
- [BLEU (Papineni 2002)](/papers/bleu-papineni-2002) — el paper original, análisis exhaustivo.
- [fundamento ROUGE](/fundamentos/rouge-metric) — el espejo recall-oriented de BLEU, para summarization.
- [ROUGE (Lin 2004)](/papers/rouge-lin-2004) — el paper de ROUGE.
- [Image Captioning](/fundamentos/image-captioning) — la tarea donde BLEU se reusa con referencias COCO.
- [decoding strategies](/fundamentos/decoding-strategies) — greedy y beam search, el "cómo" de la generación que BLEU evalúa.
- [dominio Multimodal](/dominios/multimodal) — visión + lenguaje, hogar del image captioning.

---

**Referencia primaria:** Papineni, K., Roukos, S., Ward, T., y Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL 2002, pp. 311–318. ACL Anthology P02-1040. <https://aclanthology.org/P02-1040/>
