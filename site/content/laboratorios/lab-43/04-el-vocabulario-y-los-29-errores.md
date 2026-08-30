---
title: "El vocabulario y los 29 errores"
weight: 4
---

El sistema acierta **2471 de 2500** clips. Los 29 que falla no son una muestra de ruido: **ninguno es arbitrario**, y leerlos uno por uno reencuadra qué significa el 1,16 % de error residual.

## El resultado, y una predicción fallada

| | Valor |
|---|---|
| Accuracy medida | **98,84 %** |
| Errores | **29 de 2500** |
| Accuracy del paper (25.000 clips) | 98,0 % |
| Error estándar con n = 2500 | 0,22 % |
| Intervalo de confianza 95 % | **[98,41 %, 99,27 %]** |

Antes de correr la evaluación estimé un rango de 97,5 %–98,5 %, centrado en el paper. **El resultado quedó por encima**, y la diferencia con el 98,0 % es estadísticamente significativa: el valor del paper cae fuera del intervalo, por poco. Al 98,0 % habría 50 errores en vez de 29.

La explicación más plausible no es que el modelo sea mejor que el publicado, sino que **el mini test set no es una muestra aleatoria**. Son los 5 *primeros* clips de cada palabra —`AMERICAN_00001` … `_00005`—, no 5 sorteados de los 50. Si la numeración de LRW guarda cualquier correlación con la calidad del clip —orden de extracción de los programas, confianza del alineamiento forzado con que Chung y Zisserman segmentaron el audio de la BBC— los primeros serían sistemáticamente más limpios que el promedio. Es el sesgo típico de submuestrear "los primeros N".

## El vocabulario tiene su propio techo

Antes de mirar los errores conviene mirar las 500 clases. Un análisis morfológico sobre `label_sorted.txt`:

| Métrica | Valor |
|---|---|
| Palabras en alguna familia morfológica | **103 de 500 (20,6 %)** |
| Familias distintas | 49 |
| Pares que difieren en ≤ 2 caracteres finales | **37** |

Los pares problemáticos: `AMERICA/AMERICAN`, `ATTACK/ATTACKS`, `BENEFIT/BENEFITS`, `CHANGE/CHANGES`, `LEVEL/LEVELS`, `MINISTER/MINISTERS`, `PRICE/PRICES`, `RIGHT/RIGHTS`, `MONTH/MONTHS`. Y familias de tres o cuatro miembros: `EVERY/EVERYBODY/EVERYONE/EVERYTHING`, `STATE/STATEMENT/STATES`, `HAPPEN/HAPPENED/HAPPENING`, `LEADER/LEADERS/LEADERSHIP`.

**Por qué esto castiga al stream visual y no al de audio.** Los sufijos que separan esos pares son casi todos `/s/`, `/z/`, `/t/`, `/d/` — fricativas y oclusivas alveolares, articuladas con **la lengua contra el paladar**, dentro de la boca. Los labios apenas se mueven.

Ese es el concepto de [visema](/fundamentos/lectura-de-labios): la unidad visual mínima del habla. El inglés tiene ~44 fonemas pero solo **~10–14 visemas distinguibles**, un colapso de 3–4 a 1. `/p/`, `/b/` y `/m/` son un único visema —labios cerrados y luego apertura—, por eso *pat*, *bat* y *mat* son visualmente idénticos. Lo mismo `/t/`, `/d/`, `/n/`; y `/s/`, `/z/`.

Y el número que conecta: el stream de video del paper acierta **82,0 %**, o sea falla en el **18 %**. El vocabulario tiene un **20,6 %** de palabras con un vecino morfológico confundible. No es prueba causal —haría falta la matriz de confusión del stream visual aislado—, pero las magnitudes coinciden de una forma que sugiere que buena parte del error de lipreading **no es del modelo, es del idioma**.

Un apunte de contexto: `CAMERON`, `OBAMA`, `WESTMINSTER`, `REFERENDUM`, `SYRIA`, `GREECE`, `MIGRANTS`, `DEFICIT`. El modelo no aprendió "inglés hablado" — aprendió **el léxico del noticiero de la BBC de 2015–2016**.

## Los 29 errores

| Real | Predicho | Confianza |
|---|---|---|
| MEETING | MAKING | 0,996 |
| ELECTION | ACTION | 0,980 |
| WORLD | WHILE | 0,979 |
| SPEND | SPENT | 0,969 |
| ELECTION | ACTION | 0,961 |
| WORDS | WOULD | 0,929 |
| BORDER | IMPORTANT | 0,928 |
| POSITION | OPPOSITION | 0,927 |
| EXPECT | EXPECTED | 0,891 |
| TAKEN | TAKING | 0,890 |
| ALLOW | WITHOUT | 0,879 |
| QUESTIONS | QUESTION | 0,848 |
| COMPANY | COMPANIES | 0,782 |
| BENEFITS | BENEFIT | 0,781 |
| PLACES | PRICES | 0,759 |
| ASKED | ANSWER | 0,742 |
| WORST | WORDS | 0,741 |
| PHONE | THIRD | 0,738 |
| HAPPENED | HAPPEN | 0,642 |
| WORDS | WORST | 0,616 |
| REASON | RECENT | 0,555 |
| CONSERVATIVE | SERVICES | 0,545 |
| ASKING | ASKED | 0,541 |
| BECAUSE | BIGGEST | 0,534 |
| AMONG | SOMEONE | 0,517 |
| LEVEL | LITTLE | 0,517 |
| LONDON | LIKELY | 0,461 |
| THERE | THEIR | 0,432 |
| THERE | THINK | 0,214 |

### Clasificación

| Tipo | n | Ejemplos |
|---|---|---|
| **Homófonos perfectos** | 1 | `THERE → THEIR` |
| **Flexión morfológica** | 8 | `SPEND→SPENT`, `QUESTIONS→QUESTION`, `TAKEN→TAKING` |
| **Contención léxica** | 2 | `POSITION→OPPOSITION`, `CONSERVATIVE→SERVICES` |
| **Rima o coda compartida** | 7 | `ELECTION→ACTION` (×2), `WORDS→WOULD`, `ALLOW→WITHOUT` |
| **Esqueleto consonántico** | 11 | `PHONE→THIRD`, `REASON→RECENT`, `MEETING→MAKING` |
| **Sin relación fonológica** | **0** | — |

{{< concept-alert type="clave" >}}
**Ni uno solo de los 29 errores es arbitrario.** El modelo nunca confunde `MEETING` con `SYRIA`. Cada error es un vecino fonológico de la palabra correcta.

Bajo azar, la probabilidad de que un error caiga justo sobre el pariente morfológico sería 1/499 = **0,2 %**. Observado: **27,6 %**. Un enriquecimiento de **~138×**.
{{< /concept-alert >}}

### Los casos que vale mirar de cerca

**`THERE → THEIR` (0,432) — el error irreducible.** En inglés británico son **/ðeə/** y **/ðeə/**: homófonos perfectos. Ni el audio ni el video contienen la información que los distingue; solo el contexto sintáctico podría, y el clip de 1,16 s está cortado a mitad de enunciado. **LRW incluye ambas como clases separadas, de modo que el techo teórico de la tarea es menor al 100 % por construcción del dataset.** Los dos errores de `THERE` son además las dos confianzas más bajas de toda la lista — el modelo señala su propia incertidumbre.

**`SPEND → SPENT` (0,969).** /spend/ contra /spent/. Difieren en **un solo rasgo**: la sonoridad de la consonante final. /d/ y /t/ comparten lugar de articulación, modo y **visema**, y a final de palabra el /d/ inglés se ensordece parcialmente. Con confianza 0,969.

**`PHONE → THIRD` (0,738).** Parece disparatado hasta ver la articulación: **/f/ y /θ/ comparten visema**. Ambas son fricativas sordas que exponen los dientes superiores — labiodental una, dental la otra. Es uno de los pares clásicamente inseparables en lectura de labios. Que sobreviva a la fusión con audio sugiere un clip donde el stream visual dominó la decisión.

**`REASON → RECENT` (0,555).** /ˈriːzən/ contra /ˈriːsənt/: idénticos salvo /z/ contra /s/ — otra vez el mismo visema, distinguidos solo por sonoridad.

**El nudo `W`.** `WORLD→WHILE`, `WORDS→WOULD`, `WORST→WORDS`, `WORDS→WORST`. **Cuatro de los 29 errores (14 %) viven en el clúster /wɜː-/**, y uno es un intercambio bidireccional: WORST se confunde con WORDS y WORDS con WORST.

**`ELECTION → ACTION`, dos veces, con 0,980 y 0,961.** De los 5 clips de `ELECTION`, **dos fallaron a la misma palabra con altísima confianza** — un 40 % de error en esa clase. No es ruido, es un sesgo sistemático: comparten la coda /kʃən/ y las sílabas tónicas /ˈlek/ y /ˈæk/ se parecen.

## El dato que cierra el argumento

| | Confianza media |
|---|---|
| Errores morfológicos | **0,789** |
| Todos los demás errores | 0,723 |

**El modelo no está menos seguro cuando confunde singular con plural — está *más* seguro.** No es un modelo dudando: es un modelo convencido de una respuesta que la señal no permite descartar.

Y eso reencuadra el 98,84 %. **El 1,16 % de error residual no es error de percepción.** El sistema oye y ve bien: extrae correctamente la secuencia de sonidos y de movimientos labiales. Lo que falla es la última milla, donde la información necesaria **no está en la señal**:

- la `/s/` final de un plural dura ~50 ms y cae en el borde del recorte de 1,216 s;
- la diferencia entre `/d/` y `/t/` es sonoridad pura: invisible en los labios y frágil en audio de televisión;
- `THERE` y `THEIR` son físicamente el mismo evento acústico y articulatorio.

Resolver esos casos exigiría **contexto lingüístico** — un modelo de lenguaje sobre el enunciado completo, no un clasificador sobre 1,16 s aislados. Es exactamente la brecha que separa este trabajo de 2018 de lo que vino después: [AV-HuBERT](/papers/av-hubert-shi-2022) (2022), que aprende representaciones audiovisuales autosupervisadas sobre secuencias largas y las conecta a decodificación con modelo de lenguaje; y antes [LipNet](/papers/lipnet-assael-2016) (2016), que ya trabajaba a nivel de oración con [CTC](/fundamentos/ctc-loss) — la salida de largo variable que E2E-AVSR declara como limitación pendiente.

---

**Anterior:** [La arqueología del checkpoint](03-la-arqueologia-del-checkpoint) · **Siguiente:** [Los defectos del notebook](05-los-defectos-del-notebook)
