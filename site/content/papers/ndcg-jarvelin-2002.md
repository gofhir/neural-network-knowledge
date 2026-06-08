---
title: "Cumulated Gain-Based Evaluation of IR Techniques (DCG/nDCG)"
weight: 257
math: true
---

{{< paper-card
    title="Cumulated Gain-Based Evaluation of IR Techniques"
    authors="Järvelin, Kekäläinen"
    year="2002"
    venue="ACM TOIS 2002"
    pdf="/papers/ndcg-jarvelin-2002.pdf" >}}
Este es el **paper origen de DCG y nDCG**, las métricas de ranking más usadas hoy en búsqueda, recomendación y _retrieval_. Järvelin y Kekäläinen parten de un problema concreto: precision y recall **binarias** dan el mismo crédito a un documento marginal que a uno altamente relevante, e ignoran la posición. Su propuesta combina, de forma coherente, **el grado de relevancia** y **el rango**: la ganancia acumulada (CG), su versión con **descuento logarítmico** (DCG) y la **normalización contra el ranking ideal** (nDCG) que lleva todo a la escala [0, 1].
{{< /paper-card >}}

---

## Contexto

Hacia 2002, la evaluación de [recuperación de información](/clases/clase-25) descansaba en precision y recall calculadas sobre **juicios binarios** de relevancia. TREC, el estándar de facto, aceptaba un documento como "relevante" con un umbral muy permisivo: bastaba **una sola oración** pertinente al requerimiento. Esto tiene dos consecuencias graves que el paper ataca de frente.

Primero, un documento marginalmente relevante y uno altísimamente relevante reciben **el mismo crédito**. Una técnica refinada que sabe ordenar primero el material excelente no se distingue de una mediocre que solo acierta con documentos marginales. Segundo, precision y recall son medidas de **conjunto**: no codifican que en una lista ordenada, lo que aparece arriba importa muchísimo más que lo que aparece abajo, porque el usuario es cada vez menos propenso a seguir bajando.

Los autores observan que en entornos modernos —que abruman al usuario con salidas enormes— lo único que de verdad importa es **qué tan arriba** quedan los documentos valiosos. Para sacar esas diferencias a la luz hacen falta **juicios de relevancia graduada** y una métrica que premie a las técnicas por ubicar lo altamente relevante en el tope.

Existían antecedentes —_average search length_ (Losee 1998), _expected search length_ (Cooper 1968), _normalized recall_ (Rocchio 1966), _sliding ratio_ (Pollack 1968), _satisfaction–frustration–total_ (Myaeng & Korfhage 1990), _ranked half-life_ (Borlund & Ingwersen 1998)— pero cada uno falla en algún eje: o son dicotómicos (no usan grados), o sensibles a _outliers_, o suponen que las técnicas comparadas recuperan **la misma lista** de documentos (supuesto irreal: dos técnicas distintas sobre una base grande recuperan documentos distintos, "ese es justamente el punto").

## Ideas principales

La propuesta es una familia de tres medidas que estiman la **ganancia acumulada** que el usuario obtiene al recorrer el ranking hasta cierta posición.

### Relevancia graduada, CG, DCG y nDCG

**Relevancia multinivel.** En lugar de relevante/irrelevante, se usa una escala de cuatro puntos (0–3): irrelevante (0), marginalmente relevante (1), bastante relevante (2), altamente relevante (3). La lista ordenada se convierte en un **vector de ganancia** $G$ reemplazando cada documento por su puntaje. Ejemplo del paper: $G' = \langle 3, 2, 3, 0, 0, 1, 2, 2, 3, 0, \ldots \rangle$.

**Cumulated Gain (CG).** Acumula los puntajes desde la posición 1 hasta $i$:

$$
\mathrm{CG}[i] = \begin{cases} G[1], & i = 1 \\ \mathrm{CG}[i-1] + G[i], & \text{en otro caso} \end{cases}
$$

De $G'$ se obtiene $CG' = \langle 3, 5, 8, 8, 8, 9, 11, 13, 16, \ldots \rangle$. La ganancia en el rango 7 se lee directo: 11.

**Discounted Cumulated Gain (DCG).** A mayor posición, menor el valor para el usuario. Se descuenta dividiendo por el **logaritmo del rango** —suave, no tan brusco como dividir por el rango— para permitir la persistencia del usuario. La base $b$ modela su paciencia (base baja = impaciente):

$$
\mathrm{DCG}[i] = \begin{cases} \mathrm{CG}[i], & i < b \\ \mathrm{DCG}[i-1] + G[i]/\log_b i, & i \geq b \end{cases}
$$

No se descuenta en el rango 1 (porque $\log_b 1 = 0$) ni en rangos menores que la base (daría un _boost_). Con $b=2$: $DCG' = \langle 3, 5, 6.89, 6.89, 6.89, 7.28, 7.99, 8.66, 9.61, \ldots \rangle$.

**Vector ideal e iDCG.** El mejor ranking posible coloca primero **todos** los documentos del nivel 3, luego los del nivel 2, luego los del 1, y al final los 0. Con $k, l, m$ documentos en los niveles 1, 2, 3:

$$
\mathrm{BV}[i] = \begin{cases} 3, & i \leq m \\ 2, & m < i \leq m+l \\ 1, & m+l < i \leq m+l+k \\ 0, & \text{en otro caso} \end{cases}
$$

**Normalized (D)CG.** Se divide el vector real por el ideal, componente a componente:

$$
\text{norm-vect}(V, I) = \langle v_1/i_1, v_2/i_2, \ldots, v_k/i_k \rangle
$$

El valor 1 representa desempeño ideal en esa posición; valores en $[0, 1)$ la fracción del ideal alcanzada. Punto clave: el ideal se basa en la **base de relevancia del tópico** (cuántos documentos relevantes de cada nivel existen), **no** en lo que recuperó alguna técnica. Esto distingue a nDCG del _sliding ratio_, que normaliza contra el mismo resultado y por eso depende del tamaño de la lista.

La formulación moderna que verás en la [Clase 25](/clases/clase-25) y en [ranking metrics](/fundamentos/ranking-metrics) usa la **ganancia exponencial** $2^{rel_i}-1$ (enfatiza aún más lo altamente relevante) sobre el descuento $\log_2(i+1)$, pero comparte exactamente esta estructura: DCG sobre iDCG.

## Resultados experimentales

El estudio de caso usa corridas reales del **ad hoc track de TREC-7**: colección de 528,000 documentos (1.9 GB), listas de 1000 documentos por tópico. Se tomaron **20 tópicos** y **cinco corridas** (A–E) del _manual track_, elegidos por tener juicios no binarios disponibles.

Seis estudiantes de máster rejuzgaron los documentos sobre la escala de cuatro puntos. En el subconjunto de 20 tópicos (N = 1182 relevantes): **20.1% altamente relevantes, 30.5% bastante relevantes, 49.4% marginales** — casi la mitad de los "relevantes" eran solo marginales, justo lo que motiva el paper. De los originalmente relevantes en TREC, 75% se confirmaron relevantes en algún nivel y 25% irrelevantes (los reevaluadores fueron más estrictos).

Se barrieron esquemas de pesos por nivel (0–1–1–1 binario, 0–0–0–1 solo altamente relevantes, 0–1–10–100 intermedio) y bases logarítmicas 2 y 10. Hallazgos reportados:

- El **esquema de pesos cambia el orden relativo** de las corridas: con 0–1–10–100 la corrida D aparece más efectiva que con pesos planos.
- La **distancia a la curva ideal** muestra el esfuerzo desperdiciado: en una figura hay que recuperar 30 documentos con la mejor corrida (90 con la peor) para el beneficio que idealmente daría recuperar 10.
- El **descuento estrecha** las diferencias entre sistemas; combinado con pesos no binarios, reordena cuál sistema gana.
- **Prueba estadística (Friedman):** con pesos 0–1–10–100, las corridas **D y E superan a A** (p < 0.01) en nCG y **D > A** (p < 0.05) en nDCG; con los otros esquemas la significancia desaparece.

## Limitaciones reconocibles

Los autores son explícitos sobre las debilidades:

- **Pesos de ganancia arbitrarios.** Cuantificar cuánto más vale un documento altamente relevante que uno marginal es "inherentemente arbitrario". Recomiendan probar varias cuantificaciones (planas y empinadas) como _sensitivity testing_, en lugar de un único esquema.
- **Base del descuento arbitraria.** No hay forma privilegiada de descontar; la base debe venir del escenario. Rango recomendado: 2 a 10 ($b \to 1$ es demasiado agresivo, $b \to \infty$ convierte DCG en CG).
- **La métrica no elige sus parámetros.** "La matemática funciona para cualquier combinación y no puede aconsejarnos cuál elegir": el último rango, los pesos y el descuento deben venir del escenario de uso.
- **Sin orden ni redundancia.** No manejan solapamiento entre documentos y tratan la relevancia como unidimensional cuando es multidimensional.
- **Muestra pequeña.** Los 20 tópicos son "bastante pocos para resultados confiables", aunque ilustran el comportamiento de las medidas.

## Por qué importa hoy

De la familia CG/DCG/nDCG, fue **nDCG** la que se volvió el estándar de evaluación de ranking durante las dos décadas siguientes. Su normalización a [0, 1] la hace comparable entre consultas con bases de relevancia de tamaños muy distintos, y su descuento logarítmico captura la intuición de cualquier interfaz ordenada: **lo de arriba importa más**.

Hoy nDCG@k es la métrica reportada por defecto en _learning-to-rank_, motores de búsqueda web, [sistemas de recomendación](/fundamentos/recommender-systems) y _retrieval_ para RAG. La variante exponencial $2^{rel_i}-1$ que domina la práctica actual es un descendiente directo de esta estructura original.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) (sistemas recomendadores multimodales) cierra con la sección **Metrics** y presenta nDCG de forma explícita. El laboratorio usa la formulación moderna:

$$
nDCG_p = \frac{\sum_{i=1}^{p} \frac{2^{rel_i}-1}{\log_2(i+1)}}{\sum_{i=1}^{REL_p} \frac{2^{rel_i}-1}{\log_2(i+1)}} = \frac{DCG_p}{IDCG_p}
$$

Esto es exactamente el descendiente del paper de Järvelin y Kekäläinen: numerador = DCG (ganancia descontada por el log de la posición), denominador = iDCG (lo mismo sobre el ranking ideal), cociente = nDCG en [0, 1]. El descuento $\log_2(i+1)$ es la elección de base 2 del paper (usuario impaciente).

**Ejemplo numérico de la clase.** La slide reproduce un cálculo sobre un ranking del tipo $[\text{rel}, \text{no}, \text{rel}, \text{no}, \ldots]$ con cinco documentos relevantes esparcidos en posiciones bajas:

- **DCG ≈ 1.4485** — ganancia descontada del ranking real, con los relevantes lejos del tope.
- **iDCG ≈ 2.9485** — ganancia del ranking ideal. Es verificable: cinco relevantes binarios en las cinco primeras posiciones, con descuento $1/\log_2(1+i)$, dan
$$
\frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} \approx 1 + 0.631 + 0.5 + 0.431 + 0.387 \approx 2.9485.
$$
- **nDCG = DCG/iDCG ≈ 1.4485 / 2.9485 ≈ 0.4912** — el ranking real captura cerca del **49%** de la ganancia del orden perfecto.

La lectura: un nDCG de ~0.49 dice que el sistema acertó con los documentos relevantes pero los puso demasiado abajo; el orden ideal los pondría en el tope (nDCG = 1). Es la "distancia a la curva ideal" del paper original, condensada en un solo número por consulta. Ver también [ranking metrics](/fundamentos/ranking-metrics) para el contexto de las métricas de evaluación.

## Notas y enlaces

- **Venue:** ACM Transactions on Information Systems, Vol. 20, No. 4, October 2002, pp. 422–446.
- **Sin arXiv** (revista ACM previa a la era de preprints en el área).
- Antecedente de los propios autores: Järvelin & Kekäläinen (2000), donde CG y DCG aparecen por primera vez; aplicado luego en el TREC Web Track 2001 (Voorhees).
- Cross-links: [Clase 25](/clases/clase-25) · [Ranking metrics](/fundamentos/ranking-metrics) · [Recommender systems](/fundamentos/recommender-systems).
