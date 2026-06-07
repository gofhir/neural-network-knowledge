---
title: "Fuzzy String Matching"
weight: 100
math: true
---

El **fuzzy string matching** (emparejamiento aproximado de cadenas) es el conjunto de técnicas que deciden si dos strings son "lo suficientemente parecidos" como para considerarse el mismo objeto, aun cuando **no son idénticos carácter a carácter**. Es la herramienta básica para lidiar con texto ruidoso: salidas de OCR con errores, typos humanos, variantes de un mismo nombre, transliteraciones, abreviaturas y diferencias de mayúsculas o puntuación.

El núcleo matemático es una **función de distancia (o de similaridad) entre strings**. La más célebre es la **distancia de edición de Levenshtein**, que cuenta cuántas operaciones elementales hacen falta para transformar un string en otro. Sobre ella se construyen ratios normalizados (como los de `fuzzywuzzy`/`thefuzz`), variantes que ignoran orden o duplicados, y toda la disciplina de **record linkage / entity resolution**: emparejar registros de personas o entidades a partir de campos con errores tipográficos. En el dominio clínico, esto es exactamente el problema del **patient matching** en MDM (Master Data Management) sobre datos FHIR.

Este fundamento cubre la distancia de Levenshtein y su programación dinámica, el ratio de similaridad de `fuzzywuzzy` y sus variantes, el trade-off precision/recall del threshold, otras métricas de similaridad (Jaro-Winkler, Jaccard sobre n-gramas, Hamming, fonéticas), el patrón de record linkage, y la conexión directa con el [Laboratorio 21](/laboratorios/lab-21).

---

## 1. Motivación: por qué el match exacto falla

El emparejamiento exacto (`a == b`) es frágil ante cualquier ruido. En el [Laboratorio 21](/laboratorios/lab-21), tras correr ABCNet sobre imágenes de escena, el reconocedor de texto produce lecturas que **no siempre son perfectas**. Buscar la marca `"nestle"` con un filtro exacto deja fuera todas las lecturas degradadas:

| Lectura del OCR | `== "nestle"` | ¿Es realmente "nestle"? |
|---|---|---|
| `nestle` | ✓ | sí |
| `nestl` | ✗ | sí (falta una letra) |
| `nesle` | ✗ | sí (falta una letra) |
| `nestlé` | ✗ | sí (acento) |
| `nestte` | ✗ | sí (sustitución t↔l) |
| `castle` | ✗ | no (palabra distinta) |

El match exacto comete un **falso negativo** en cada fila intermedia: descarta instancias que sí corresponden a la marca buscada. Lo mismo ocurre con typos humanos (`recieve` vs `receive`), variantes de nombres (`Jon` / `John` / `Jonh`), o formatos de direcciones (`Av.` / `Avenida`).

La solución es relajar la igualdad a una **noción de cercanía**: definimos una distancia $d(s, t)$ entre dos strings y aceptamos como match todo par con $d$ pequeña (o, equivalentemente, con similaridad alta sobre un umbral). El problema se traslada entonces a **(a)** elegir una buena métrica y **(b)** elegir el umbral.

{{< concept-alert type="clave" >}}
Fuzzy matching cambia una decisión binaria exacta por una decisión basada en un **score continuo más un threshold**. Eso introduce inevitablemente el dilema **precisión vs recall**: cuanto más permisivo el umbral, más verdaderos positivos rescatás pero más falsos positivos arrastrás.
{{< /concept-alert >}}

---

## 2. Distancia de Levenshtein

### 2.1 Definición formal

Dadas dos cadenas $s$ de longitud $m$ y $t$ de longitud $n$, la **distancia de Levenshtein** $\operatorname{lev}(s, t)$ es el **mínimo número de operaciones de edición de un solo carácter** necesarias para transformar $s$ en $t$, donde las operaciones permitidas son:

- **Inserción** de un carácter (costo 1).
- **Eliminación** de un carácter (costo 1).
- **Sustitución** de un carácter por otro (costo 1).

Por ejemplo, $\operatorname{lev}(\text{"nestle"}, \text{"nestl"}) = 1$ (una eliminación de la `e` final) y $\operatorname{lev}(\text{"kitten"}, \text{"sitting"}) = 3$ (sustituir `k`→`s`, sustituir `e`→`i`, insertar `g`).

Es una **métrica** en sentido estricto: $\operatorname{lev}(s,t) \geq 0$, $\operatorname{lev}(s,t)=0 \iff s=t$, es simétrica, y cumple la desigualdad triangular. Eso la hace apta para indexación métrica (BK-trees, VP-trees) en búsquedas a gran escala.

### 2.2 La recurrencia de programación dinámica

El cálculo eficiente se hace con programación dinámica. Sea $d(i, j)$ la distancia de edición entre el prefijo $s_{1:i}$ (primeros $i$ caracteres de $s$) y el prefijo $t_{1:j}$. La recurrencia es:

{{< math-formula title="Recurrencia de Levenshtein" >}}
d(i,j) = \min \begin{cases}
d(i-1,\, j) + 1 & \text{(eliminación de } s_i) \\
d(i,\, j-1) + 1 & \text{(inserción de } t_j) \\
d(i-1,\, j-1) + [\,s_i \neq t_j\,] & \text{(sustitución / coincidencia)}
\end{cases}
{{< /math-formula >}}

donde $[\,s_i \neq t_j\,]$ es la **indicatriz de Iverson**: vale $0$ si los caracteres coinciden (no se paga nada, es un "match") y $1$ si difieren (sustitución). Las **condiciones de borde** capturan transformar un prefijo desde/hacia el string vacío:

$$d(i, 0) = i \quad (\text{borrar } i \text{ caracteres}), \qquad d(0, j) = j \quad (\text{insertar } j \text{ caracteres}).$$

El resultado final es $\operatorname{lev}(s, t) = d(m, n)$.

### 2.3 La matriz $(m+1) \times (n+1)$

La DP se materializa como una matriz de $(m+1) \times (n+1)$ celdas: una fila/columna extra para el prefijo vacío. Cada celda $d(i,j)$ se llena a partir de sus tres vecinas (arriba, izquierda, diagonal superior-izquierda). Ejemplo completo para `s = "nestle"`, `t = "nestl"`:

```text
        ""  n   e   s   t   l
   ""    0   1   2   3   4   5
   n     1   0   1   2   3   4
   e     2   1   0   1   2   3
   s     3   2   1   0   1   2
   t     4   3   2   1   0   1
   l     5   4   3   2   1   0
   e     6   5   4   3   2   1   <- lev = 1
```

La esquina inferior-derecha vale **1**: una sola edición separa `"nestle"` de `"nestl"`. El **camino de retroceso** (backtracking) por la matriz reconstruye la secuencia de operaciones óptima, lo que conecta directamente con el **alineamiento de secuencias**.

### 2.4 Complejidad y conexión con alineamiento

- **Tiempo**: $O(m \cdot n)$ — una pasada por cada celda.
- **Memoria**: $O(m \cdot n)$ ingenuamente, pero **$O(\min(m, n))$** si solo se necesita el valor (basta mantener dos filas, porque la recurrencia solo mira la fila anterior).

La estructura es **idéntica al alineamiento global de Needleman-Wunsch** usado en bioinformática para alinear secuencias de ADN/proteínas: la distancia de Levenshtein es el caso particular con costo unitario de match/mismatch/gap. La variante **Damerau-Levenshtein** añade una cuarta operación, la **transposición** de dos caracteres adyacentes (`ab`→`ba`, costo 1), que modela bien los typos de teclado (`teh`→`the`). Y el **alineamiento local de Smith-Waterman** es el análogo que busca la mejor subcadena en lugar del string completo.

{{< concept-alert type="recordar" >}}
La distancia de Levenshtein **NO está normalizada**: $\operatorname{lev}=2$ es catastrófico entre dos strings de largo 3, pero trivial entre dos strings de largo 100. Para comparar pares de longitudes distintas hay que **normalizar** — eso es justamente lo que hace el ratio de similaridad de la sección siguiente.
{{< /concept-alert >}}

---

## 3. Ratio de similaridad (fuzzywuzzy / thefuzz)

`fuzzywuzzy` (hoy renombrada **`thefuzz`**) convierte la distancia de edición en una **similaridad normalizada a $[0, 100]$**, donde 100 significa idénticos y 0 significa sin nada en común. La fórmula básica de `ratio` se apoya en el algoritmo de Ratcliff-Obershelp (vía `difflib.SequenceMatcher` de la librería estándar), que cuenta los caracteres en común mediante el matching de subsecuencias:

{{< math-formula title="Ratio de similaridad" >}}
\text{ratio} = \frac{2 \cdot M}{T} \times 100
{{< /math-formula >}}

donde $T$ es la **suma de las longitudes** de ambos strings ($T = |s| + |t|$) y $M$ es el **número de caracteres coincidentes** (los matches encontrados por el matcher de subsecuencias). El factor $2$ y el denominador $T$ hacen que el resultado sea simétrico y caiga en $[0, 100]$.

Para `"nestle"` (6) vs `"nestl"` (5): $T = 11$, $M = 5$ (los 5 caracteres `n,e,s,t,l` coinciden), entonces

$$\text{ratio} = \frac{2 \cdot 5}{11} \times 100 \approx 90.9 \rightarrow 91.$$

Un ratio de 91 supera holgadamente el threshold de 80 usado en el laboratorio, por lo que `"nestl"` se acepta como match de `"nestle"`.

{{< concept-alert type="ojo" >}}
El `ratio` de `fuzzywuzzy` **no es exactamente** $100 \cdot (1 - \operatorname{lev}/\max(m,n))$. Internamente usa Ratcliff-Obershelp (matching de subsecuencias comunes), no Levenshtein puro. Son métricas distintas que correlacionan fuerte pero pueden diferir en casos borde. El extra `[speedup]` instala `python-Levenshtein`, que reemplaza el backend por una implementación **en C basada en Levenshtein real**, mucho más rápida — y eso puede cambiar levemente los scores respecto al backend Python por defecto.
{{< /concept-alert >}}

### 3.1 Las cuatro variantes principales

`fuzzywuzzy`/`thefuzz` ofrecen cuatro funciones, cada una pensada para un tipo de ruido distinto:

| Función | Qué hace | Cuándo usarla |
|---|---|---|
| **`ratio`** | Similaridad directa entre los dos strings completos | Strings cortos y comparables (palabra vs palabra) |
| **`partial_ratio`** | Mejor match de la cadena corta como **subcadena** de la larga | Cuando uno está contenido en el otro (`"nestle"` en `"nestle chocolate bar"`) |
| **`token_sort_ratio`** | Tokeniza, **ordena alfabéticamente** los tokens y luego compara | Cuando el **orden de las palabras** cambia (`"juan perez"` vs `"perez juan"`) |
| **`token_set_ratio`** | Tokeniza, toma la **intersección y diferencias de conjuntos** de tokens, ignora orden **y duplicados** | Cuando hay palabras extra o repetidas (`"nestle"` vs `"the nestle nestle company"`) |

Reglas prácticas:

- **`ratio`** es el default para comparar términos atómicos (una marca, una palabra de OCR). Es lo que usa el Lab 21.
- **`partial_ratio`** rescata el caso "la keyword aparece dentro de un texto más largo": el cartel dice `"PHARMACY OPEN 24H"` y buscás `"pharmacy"`.
- **`token_sort_ratio`** neutraliza diferencias de orden, típico en nombres de personas (`"García López, Ana"` vs `"Ana García López"`).
- **`token_set_ratio`** es el más permisivo: ignora orden y duplicados, ideal cuando un campo tiene tokens adicionales (títulos, sufijos) que el otro no.

```python
from thefuzz import fuzz   # antes: from fuzzywuzzy import fuzz

fuzz.ratio("nestle", "nestl")                 # 91  -> match (> 80)
fuzz.ratio("nestle", "nesle")                 # 91  -> match
fuzz.ratio("nestle", "castle")                # 67  -> NO match (< 80)

fuzz.partial_ratio("pharmacy", "pharmacy 24h open")   # 100 (subcadena exacta)
fuzz.token_sort_ratio("ana garcia lopez",
                      "lopez ana garcia")             # 100 (mismo conjunto, otro orden)
fuzz.token_set_ratio("nestle",
                     "the nestle nestle company")     # 100 (ignora extras y duplicados)
```

Existe también `WRatio`, una heurística que combina las variantes anteriores y elige automáticamente según las longitudes relativas — útil cuando no se sabe de antemano qué tipo de discrepancia esperar. Para búsqueda 1-contra-muchos, el submódulo `process` (`process.extractOne`, `process.extract`) aplica cualquiera de estas funciones contra una lista completa y devuelve los mejores candidatos.

---

## 4. El trade-off precision/recall del threshold

Aceptar un par como match cuando `score > threshold` es un **clasificador binario** sobre el score continuo. Mover el umbral desliza el punto de operación a lo largo de la curva precision-recall:

- **Threshold alto** (p. ej. 95): solo se aceptan pares casi idénticos. **Alta precisión** (casi todos los aceptados son verdaderos matches) pero **bajo recall** (se escapan las variantes ruidosas: `"nestl"`, `"nesle"` quedarían fuera si el umbral fuera 92).
- **Threshold bajo** (p. ej. 60): se aceptan pares lejanos. **Alto recall** (se capturan casi todas las variantes verdaderas) pero **baja precisión** (entran falsos positivos: `"castle"` con ratio 67 se colaría).

Definiendo, sobre el conjunto de pares aceptados, las cantidades usuales:

$$\text{precisión} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN},$$

el umbral óptimo depende del **costo asimétrico de cada tipo de error**. La pregunta operativa es: ¿qué duele más, un falso positivo o un falso negativo?

- En **búsqueda exploratoria** (encontrar todas las menciones de una marca en un dataset), un falso positivo es barato — el humano lo descarta de un vistazo — y un falso negativo es caro porque pierde información. Conviene **threshold bajo** (alto recall).
- En **fusión irreversible de registros** (unir dos historias clínicas en MDM), un falso positivo es catastrófico (mezclás dos pacientes distintos) y un falso negativo es recuperable (queda un duplicado que se puede limpiar después). Conviene **threshold alto** (alta precisión), o derivar los casos dudosos a revisión manual.

El valor **80** del Lab 21 es un punto medio sensato para búsqueda de keywords sobre OCR: suficientemente bajo como para tolerar 1-2 errores de lectura, suficientemente alto como para no confundir palabras distintas.

{{< concept-alert type="clave" >}}
No existe "el threshold correcto" universal. Se calibra con un **conjunto de validación etiquetado** (pares con label match/no-match), maximizando la métrica que refleje el costo real del negocio: F1 si los errores cuestan parecido, F$_\beta$ con $\beta > 1$ si los falsos negativos duelen más, o precisión-a-recall-fijo si hay un SLA de cobertura.
{{< /concept-alert >}}

---

## 5. Otras métricas de similaridad de strings

Levenshtein y el ratio no son las únicas herramientas. Según la naturaleza del ruido conviene una métrica distinta:

| Métrica | Idea | Mejor para |
|---|---|---|
| **Jaro-Winkler** | Similaridad de Jaro (matches dentro de una ventana + transposiciones) con **bonus a prefijos comunes** | **Nombres propios y apellidos cortos**; el bonus de prefijo refleja que los nombres rara vez difieren al inicio |
| **Jaccard sobre n-gramas** | $J = \dfrac{|A \cap B|}{\|A \cup B|}$ sobre el conjunto de n-gramas (trigramas) de cada string | **Textos largos**, robusto a reordenamientos; base de búsqueda fuzzy en bases de datos (índices GIN/trigram de PostgreSQL) |
| **Distancia de Hamming** | Número de posiciones en que difieren dos strings **de igual longitud** | Códigos de longitud fija (DNI, códigos de barras, identificadores), no maneja inserciones/eliminaciones |
| **Soundex / Metaphone / NYSIIS** | Codifican la **pronunciación** y comparan los códigos fonéticos | Variantes que **suenan igual** (`Smith` / `Smyth`, `Catalina` / `Katalina`); útil en transcripción de nombres dictados |
| **Cosine de TF-IDF / embeddings** | Vectoriza y compara por coseno | Similaridad **semántica** (no léxica): `auto` ≈ `vehículo`, fuera del alcance de la edición de caracteres |

La **distancia de Jaro** se define formalmente como

$$\operatorname{jaro}(s,t) = \begin{cases} 0 & \text{si } m = 0 \\ \dfrac{1}{3}\left( \dfrac{m}{|s|} + \dfrac{m}{|t|} + \dfrac{m - \tau}{m} \right) & \text{si } m > 0 \end{cases}$$

donde $m$ es el número de caracteres coincidentes (dentro de una ventana de $\lfloor \max(|s|,|t|)/2 \rfloor - 1$) y $\tau$ es el número de transposiciones. **Jaro-Winkler** añade $\ell \cdot p \cdot (1 - \operatorname{jaro})$, con $\ell$ el largo del prefijo común (hasta 4) y $p \approx 0.1$ el factor de escala. En la práctica, **Jaro-Winkler es el estándar de facto para emparejar nombres de personas** en record linkage, mientras que **Levenshtein/ratio sirve mejor para palabras de OCR** y **Jaccard sobre trigramas para textos largos o búsqueda en BD**.

---

## 6. Record linkage / entity resolution

El **record linkage** (también **entity resolution** o **data matching**) es la aplicación clásica del fuzzy matching: decidir si dos registros de distintas fuentes (o de la misma fuente con duplicados) se refieren a la **misma entidad del mundo real** — la misma persona, empresa, dirección o producto — pese a errores tipográficos, formatos distintos y campos faltantes.

### 6.1 El patrón blocking + scoring

Comparar todos los pares de $N$ registros es $O(N^2)$, inviable para millones de registros. El pipeline canónico tiene dos fases:

1. **Blocking (indexing)**: agrupar candidatos plausibles por una clave barata (p. ej. mismo código postal, o mismo primer trigrama del apellido + año de nacimiento). Solo se comparan pares **dentro del mismo bloque**, reduciendo drásticamente el número de comparaciones. El blocking prioriza **recall**: nunca debe descartar un par que en verdad matchea.

2. **Scoring (matching)**: para cada par candidato sobreviviente, calcular un **score de similaridad por campo** (Jaro-Winkler sobre el nombre, Levenshtein sobre la dirección, igualdad exacta sobre la fecha de nacimiento) y **combinarlos** en una decisión final. El scoring prioriza **precisión**.

El modelo clásico de combinación es el de **Fellegi-Sunter** (1969), que pondera cada campo según su poder discriminativo (un match en `apellido` informa más que un match en `ciudad`) mediante pesos derivados de probabilidades de acuerdo/desacuerdo, y produce un score log-likelihood comparado contra dos umbrales (match automático / no-match automático / **zona gris** que va a revisión humana). Variantes modernas reemplazan los pesos manuales por un **clasificador entrenado** (regresión logística, o un GBM tipo XGBoost) sobre los scores por campo.

### 6.2 Por qué combinar múltiples campos

Ningún campo aislado basta. Dos personas pueden compartir nombre (`Juan Pérez` hay miles), o una misma persona tener su nombre escrito de tres formas. La clave es que **combinar campos sube la precisión sin sacrificar recall**: un match difuso en el nombre **más** coincidencia exacta de fecha de nacimiento **más** match difuso en identificador es evidencia mucho más fuerte que cualquiera por separado. Cada campo aporta independencia condicional.

En el **patient matching de datos FHIR** (MDM clínico), los campos típicos son `name.family`, `name.given`, `birthDate`, `gender`, `identifier` (RUT/DNI), `address` y `telecom`. Un buen matcher pondera la **especificidad** de cada uno: dos pacientes con el mismo `identifier` casi seguro son la misma persona (alta especificidad), mientras que coincidir en `gender` no dice casi nada. La combinación es lo que distingue un MDM serio de un `LIKE '%nombre%'`.

### 6.3 El peligro de los dos errores

En MDM clínico los dos errores tienen **costos clínicos asimétricos y graves**:

- **Falso positivo (over-merge)**: fusionar dos pacientes **distintos** en un solo registro. Es el error más peligroso: mezcla historias clínicas, alergias, medicaciones y resultados de laboratorio de dos personas. Puede causar daño directo al paciente (administrar un fármaco al que el "otro" paciente es alérgico). Suele ser **difícil de revertir** una vez propagado.
- **Falso negativo (under-merge / duplicado)**: no reconocer que dos registros son el mismo paciente, creando un duplicado. Fragmenta la historia clínica (un médico no ve resultados que existen en el otro registro), pero es **recuperable**: un proceso de deduplicación posterior puede unirlos.

Por eso el patient matching clínico opera con **umbral alto** (favorece precisión), envía la **zona gris a revisión humana** (steward de datos), y prefiere dejar un duplicado antes que arriesgar una fusión incorrecta. Es el espejo exacto del trade-off de la sección 4, llevado a un dominio donde un falso positivo puede tener consecuencias clínicas.

---

## 7. Implementación didáctica de Levenshtein

Una implementación de la matriz DP completa con NumPy, que devuelve la distancia y permite inspeccionar la tabla:

```python
import numpy as np

def levenshtein(s: str, t: str) -> int:
    m, n = len(s), len(t)
    d = np.zeros((m + 1, n + 1), dtype=int)
    # Condiciones de borde: transformar desde/hacia el string vacío
    d[:, 0] = np.arange(m + 1)   # d(i, 0) = i  (i eliminaciones)
    d[0, :] = np.arange(n + 1)   # d(0, j) = j  (j inserciones)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1   # [s_i != t_j]
            d[i, j] = min(
                d[i - 1, j] + 1,         # eliminación
                d[i, j - 1] + 1,         # inserción
                d[i - 1, j - 1] + cost,  # sustitución / match
            )
    return int(d[m, n])

assert levenshtein("nestle", "nestl") == 1
assert levenshtein("kitten", "sitting") == 3
assert levenshtein("nestle", "castle") == 3
```

Y la versión que usa el laboratorio, con `thefuzz` y su ratio normalizado:

```python
from thefuzz import fuzz

keyword = "nestle"
threshold = 80          # ratio mínimo aceptado (0 a 100)

predicciones_ocr = ["nestle", "nestl", "nesle", "castle", "nestte"]
matches = [w for w in predicciones_ocr if fuzz.ratio(w, keyword) > threshold]
# -> ['nestle', 'nestl', 'nesle', 'nestte']   ('castle' queda fuera con ratio 67)
```

---

## 8. Conexión con el Laboratorio 21

En el [Laboratorio 21](/laboratorios/lab-21) (Scene Text Recognition con ABCNet), el fuzzy matching es el **puente entre el OCR ruidoso y la búsqueda por keyword**:

1. **Búsqueda de `"nestle"`**. El conteo exacto sobre el diccionario de frecuencias encuentra solo las lecturas perfectas. Aplicando luego `fuzz.ratio(word, "nestle")` con `threshold = 80`, el laboratorio **rescata 3 instancias adicionales** que el match exacto había dejado pasar — lecturas con uno o dos errores de OCR que igualmente corresponden a la marca. El propio notebook concluye: "El valor del threshold va a depender de qué tan importante sea capturar todas las instancias versus qué tan malo es agregar resultados incorrectos. Mientras más bajo el threshold, más probable es agregar lecturas indeseadas" — que es exactamente el trade-off precision/recall de la sección 4.

2. **`draw_in_map(keyword, ..., threshold=80)`**. La función que dibuja las detecciones sobre un mapa usa el **mismo patrón**: para cada palabra leída calcula `ratio = fuzz.ratio(word, keyword)` y la marca en el mapa si `ratio > threshold`. Así se buscan keywords como **`"pharmacy"`**, **`"food"`** y **`"university"`** sobre los carteles reconocidos, tolerando los errores del reconocedor. Sin fuzzy matching, un cartel leído como `"pharmcy"` o `"phamacy"` nunca aparecería en el mapa.

El laboratorio es, en miniatura, un sistema de **information retrieval sobre texto ruidoso**: el OCR genera el "texto" y el fuzzy matching hace la recuperación tolerante a errores. La misma maquinaria — distancia de Levenshtein, ratio normalizado, threshold calibrado — escala directamente al **record linkage** de datos estructurados (sección 6).

---

## 9. Resumen

- El **match exacto falla con texto ruidoso** (OCR, typos, variantes). El fuzzy matching lo reemplaza por un **score de similaridad + threshold**.
- La **distancia de Levenshtein** cuenta inserciones, eliminaciones y sustituciones mínimas; se calcula con una **matriz DP $(m+1)\times(n+1)$** en $O(m\cdot n)$, y es el caso unitario del alineamiento de secuencias (Needleman-Wunsch).
- El **ratio de `fuzzywuzzy`/`thefuzz`** normaliza a $[0,100]$ con $\text{ratio} = 2M/T \times 100$. Las variantes `partial_ratio`, `token_sort_ratio` y `token_set_ratio` manejan subcadenas, reordenamientos y duplicados. El extra `[speedup]` usa `python-Levenshtein` en C.
- El **threshold** materializa el dilema **precisión vs recall**: alto → precisión, bajo → recall. Se calibra según el **costo de cada tipo de error**.
- Otras métricas: **Jaro-Winkler** (nombres), **Jaccard sobre n-gramas** (textos largos / BD), **Hamming** (longitud fija), **fonéticas** (Soundex/Metaphone).
- El **record linkage / entity resolution** aplica todo esto con el patrón **blocking + scoring**, combinando **múltiples campos** para subir precisión sin perder recall. En **MDM clínico FHIR**, el **falso positivo (over-merge)** es el error peligroso e irreversible; el falso negativo (duplicado) es recuperable.
- En el **Lab 21**, `fuzz.ratio` con threshold 80 rescata 3 lecturas imperfectas de `"nestle"` y alimenta `draw_in_map` para buscar `"pharmacy"`/`"food"`/`"university"` sobre OCR ruidoso.

Ver también: [CTC Loss](/fundamentos/ctc-loss) · [ROUGE Metric](/fundamentos/rouge-metric) · [Scene Text Recognition](/fundamentos/scene-text-recognition) · [Laboratorio 21](/laboratorios/lab-21).
