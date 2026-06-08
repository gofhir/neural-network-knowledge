# Cumulated Gain-Based Evaluation of IR Techniques

**Kalervo Järvelin, Jaana Kekäläinen** — University of Tampere
**ACM Transactions on Information Systems (TOIS)**, Vol. 20, No. 4, October 2002, pp. 422–446.
Recibido enero 2002; revisado julio 2002; aceptado septiembre 2002.

---

## Contexto: por qué precision/recall binarias se quedaron cortas

El paper arranca de una observación que en 2002 ya era incómoda y hoy es evidente: los entornos de recuperación de información (IR) modernos abruman al usuario con salidas enormes. Si una consulta devuelve miles de documentos, lo único que importa en la práctica es **qué tan arriba** quedan los documentos verdaderamente valiosos. Sin embargo, las dos métricas dominantes de la época —precision y recall— eran ciegas a esto por dos limitaciones estructurales.

**Primera limitación: relevancia binaria.** La práctica de laboratorio (y en particular TREC) asignaba a cada documento un juicio binario de relevancia tópica con un umbral muy permisivo: bastaba que el documento tuviera **al menos una oración** pertinente al requerimiento para contar como "relevante" (cita explícita a TREC 2001 en el paper). Bajo este criterio, un documento marginalmente relevante y uno altísimamente relevante reciben **el mismo crédito**. Como consecuencia, una técnica de recuperación sofisticada que sabe ordenar primero los documentos excelentes no se distingue de una técnica mediocre que solo acierta con material marginal. Los autores lo dicen sin rodeos: las diferencias entre técnicas "sloppy" y "excellent" respecto a documentos altamente relevantes no se vuelven visibles en la evaluación. Para sacar esas diferencias a la luz hacen falta **juicios de relevancia graduada** y un método para usarlos.

**Segunda limitación: insensibilidad a la posición.** Precision y recall, calculadas sobre conjuntos, no codifican directamente el orden del ranking. Las curvas P–R atenúan esto, pero —como argumentan los autores en la discusión— ocultan cuántos documentos hay que examinar para alcanzar cierto nivel de recall, y "enmascaran el mal desempeño" (cita a Losee 1998). Desde el punto de vista del usuario, lo que cuenta es la ganancia acumulada **a medida que recorre la lista de arriba hacia abajo**, no un agregado de conjunto.

El paper sitúa su propuesta dentro de una genealogía: ya existían medidas que tocaban estos puntos —*average search length* (ASL, Losee 1998), *expected search length* (ESL, Cooper 1968), *normalized recall* (NR, Rocchio 1966; Salton & McGill 1983), *sliding ratio* (SR, Pollack 1968; Korfhage 1997), *satisfaction–frustration–total* (SFT, Myaeng & Korfhage 1990) y *ranked half-life* (RHL, Borlund & Ingwersen 1998)—, pero cada una falla en algún eje: ASL/ESL/NR/RHL son dicotómicas (no usan grados de relevancia) o sensibles a outliers; SR y SFT sí usan grados pero **suponen que las técnicas comparadas recuperan la misma lista de documentos**, supuesto irreal porque dos técnicas distintas, sobre una base de N documentos con n ≪ N, recuperan documentos muy distintos —"that is the whole idea (!)", remarca el paper.

## Contribución: tres medidas — CG, DCG, nDCG (más la normalización por el ideal)

La contribución central es una familia de métricas que estiman la **ganancia acumulada (cumulated gain)** que el usuario obtiene al examinar el resultado hasta cierta posición del ranking, combinando de forma coherente **el grado de relevancia** y **la posición**:

1. **Cumulated Gain (CG):** acumula los puntajes de relevancia a lo largo de la lista ordenada.
2. **Discounted Cumulated Gain (DCG):** igual que CG, pero aplica un **factor de descuento logarítmico** que penaliza progresivamente a los documentos recuperados tarde, modelando que el usuario es cada vez menos propenso a seguir bajando.
3. **Normalized (D)CG — nCG y nDCG:** divide el vector (D)CG real por el (D)CG **ideal** (el mejor ranking posible para esa consulta), llevando todo a la escala [0, 1] donde 1 representa el desempeño perfecto.

La clave conceptual: la normalización es contra **un ideal basado en la base de relevancia del tópico** (cuántos documentos relevantes de cada nivel existen realmente), no contra el resultado recuperado por alguna técnica. Eso distingue a nDCG del *sliding ratio*, que normaliza contra el mismo resultado recuperado y por ende es sensible al tamaño de la lista.

## Método (grounded en las ecuaciones del paper)

### Relevancia multinivel

Los autores asumen una escala de cuatro puntos (0 a 3), donde 3 denota alto valor y 0 ningún valor. La lista ordenada de documentos se convierte en un **vector de ganancia** G reemplazando cada ID de documento por su puntaje de relevancia. Ejemplo del paper:

```
G' = ⟨3, 2, 3, 0, 0, 1, 2, 2, 3, 0, ...⟩
```

La escala de cuatro niveles usada en el estudio de caso es: (1) **irrelevante** — no contiene información sobre el tópico; (2) **marginalmente relevante** — solo apunta al tópico; (3) **bastante relevante** (*fairly relevant*) — contiene más información que el enunciado del tópico pero no es exhaustivo; (4) **altamente relevante** (*highly relevant*) — discute los temas del tópico exhaustivamente.

### Cumulated Gain (CG)

El CG en la posición *i* se computa sumando desde la posición 1 hasta *i*. Definición recursiva (Ec. 1 del paper), con G[i] el puntaje en la posición *i*:

$$
\mathrm{CG}[i] = \begin{cases} G[1], & \text{si } i = 1 \\ \mathrm{CG}[i-1] + G[i], & \text{en otro caso} \end{cases}
$$

A partir de G' el paper obtiene CG' = ⟨3, 5, 8, 8, 8, 9, 11, 13, 16, 16, ...⟩. La ganancia acumulada en cualquier rango se lee directamente: en el rango 7 vale 11.

### Discounted Cumulated Gain (DCG)

El descuento parte del segundo principio: a mayor posición, menos valioso es el documento para el usuario. Se necesita una función de descuento que reduzca el puntaje a medida que sube el rango, **pero no demasiado bruscamente** (no como dividir por el rango), para permitir la persistencia del usuario. La solución elegante: **dividir por el logaritmo del rango**. Por ejemplo, $\log_2 2 = 1$ y $\log_2 1024 = 10$, de modo que un documento en la posición 1024 todavía conserva un décimo de su valor nominal. La base del logaritmo *b* modela el comportamiento del usuario: base baja (b=2) modela un usuario impaciente; base alta (b>10) modela un usuario paciente. Definición (Ec. 2 del paper):

$$
\mathrm{DCG}[i] = \begin{cases} \mathrm{CG}[i], & \text{si } i < b \\ \mathrm{DCG}[i-1] + G[i]/\log_b i, & \text{si } i \geq b \end{cases}
$$

Dos detalles finos que el paper enfatiza: (a) **no se aplica el descuento en el rango 1** porque $\log_b 1 = 0$; (b) **no se aplica el caso de descuento para rangos menores que la base** del logaritmo, porque dividir por un log < 1 daría un *boost* en lugar de un descuento. Esto es realista: cuanto mayor la base, menor el descuento y más probable que el usuario examine al menos hasta el rango base. Con b=2, de G' el paper obtiene DCG' = ⟨3, 5, 6.89, 6.89, 6.89, 7.28, 7.99, 8.66, 9.61, 9.61, ...⟩.

En la discusión los autores generalizan el descuento como $\mathrm{DCG}[i] = \mathrm{DCG}[i-1] + G[i]/df$, con tres casos: si **df = 1** entonces DCG = CG (sin descuento); si **df = i** (división por el rango) el descuento es demasiado agudo y poco realista; si **df = $\log_b i$** el descuento es suave y ajustable. Concluyen que el rango de bases 2 a 10 cubre bien la mayoría de escenarios.

### Vector ideal y normalización (iDCG)

El **vector de mejor puntaje posible** BV (Ec. 4) se construye colocando primero todos los documentos del nivel 3, luego los del nivel 2, luego los del 1 y finalmente los 0. Formalmente, si hay *k*, *l*, *m* documentos relevantes en los niveles 1, 2 y 3 respectivamente:

$$
\mathrm{BV}[i] = \begin{cases} 3, & \text{si } i \leq m \\ 2, & \text{si } m < i \leq m+l \\ 1, & \text{si } m+l < i \leq m+l+k \\ 0, & \text{en otro caso} \end{cases}
$$

Ejemplo del paper: I' = ⟨3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, ...⟩, del cual se obtienen los vectores ideales CG'_I = ⟨3, 6, 9, 11, 13, 15, 16, 17, 18, 19, 19, 19, 19, ...⟩ y DCG'_I = ⟨3, 6, 7.89, 8.89, 9.75, 10.52, 10.88, 11.21, 11.53, 11.83, 11.83, ...⟩ (con b=2).

La normalización (Ec. 5) divide componente a componente el vector real por el ideal:

$$
\text{norm-vect}(V, I) = \langle v_1/i_1, v_2/i_2, \ldots, v_k/i_k \rangle
$$

Ejemplo del paper: nCG' = ⟨1, 0.83, 0.89, 0.73, 0.62, 0.6, 0.69, 0.76, 0.89, 0.84, ...⟩. El valor 1 representa desempeño ideal en esa posición; valores en [0, 1) la fracción del ideal alcanzada. El vector ideal normalizado consigo mismo es siempre ⟨1, 1, ..., 1⟩. El paper subraya que **el vector ideal se basa en la base de recall del tópico**, no en el resultado de una técnica — diferencia esencial frente a sliding ratio y satisfaction measure (Korfhage 1997).

## Experimentos del paper (estudio de caso TREC-7)

El estudio de caso usa corridas reales del **ad hoc track de TREC-7**: la colección incluía 528,000 documentos (1.9 GB); los participantes devolvieron listas de los mejores 1000 documentos por tópico. Se usaron listas de resultados para **20 tópicos** de **cinco participantes** (corridas A–E) del *manual track*, elegidos porque tenían juicios de relevancia no binarios disponibles (Sormunen 2002).

**Rejuicio graduado.** Seis estudiantes de máster en estudios de información, fluidos en inglés, rejuzgaron los documentos previamente marcados relevantes por los evaluadores de NIST más ~5% de los irrelevantes, sobre la escala de cuatro puntos. La Tabla I del paper muestra la distribución: de los documentos originalmente "relevantes" de TREC, 75% fueron juzgados relevantes en algún nivel y 25% irrelevantes (los reevaluadores fueron más estrictos); el 93.8% de los originalmente irrelevantes se confirmaron irrelevantes. En el subconjunto de 20 tópicos (N = 1182 documentos relevantes): 20.1% altamente relevantes, 30.5% bastante relevantes, 49.4% marginales — es decir, **cerca de la mitad de los "relevantes" eran solo marginales**, justo el punto que motiva el paper.

**Parámetros barridos:** (a) esquemas de pesos por nivel de relevancia: 0–1–1–1 (binario TREC), 0–0–0–1 (solo cuentan los altamente relevantes), y 0–1–10–100 (escala intermedia donde los altamente relevantes valen 100× los marginales); (b) bases logarítmicas 2 y 10 (mostraron solo base 2 por ser la prueba más estricta y modelar usuarios impacientes).

**Curvas gain-by-rank (Figuras 1–4).** Las Figuras 1(a,b) muestran las curvas CG para las cinco corridas más la ideal; las 2(a,b) las DCG; las 3(a,b) las nCG normalizadas; las 4(a,b) las nDCG. Hallazgos cualitativos reportados: (i) el esquema de pesos **cambia el orden relativo** de las corridas — con 0–1–10–100 la corrida D parece más efectiva que con 0–1–1–1; (ii) la distancia vertical entre la curva real y la ideal muestra el "esfuerzo desperdiciado" en documentos imperfectos; (iii) interpretación orientada al usuario — en la Fig. 1(a) hay que recuperar 30 documentos con la mejor corrida (90 con la peor) para obtener el beneficio que idealmente daría recuperar solo 10; (iv) el descuento **estrecha** las diferencias entre sistemas (comparar Fig. 1 con Fig. 2), y combinado con pesos no binarios reordena los sistemas (en Fig. 2(b) la corrida A pierde y la C se beneficia).

**Prueba estadística (Tabla II).** Promedios de n(D)CG sobre los 20 tópicos por corrida, con test de Friedman. Resultados reportados: con pesos 0–1–10–100, **D y E > A** (p < 0.01) en nCG, y **D > A** (p < 0.05) en nDCG; con los otros esquemas la significancia desaparece. ANOVA no probó diferencias significativas — los autores discuten que la escala ordinal sugiere tests no paramétricos (Friedman/Wilcoxon), pero que al introducir pesos en escala de razón se justificarían tests paramétricos (ANOVA, t-test) si se cumplen los supuestos.

## Limitaciones reconocidas por los autores

El propio paper enumera debilidades, varias compartidas con las medidas tradicionales:

- **Arbitrariedad de los pesos de ganancia.** Cuantificar cuánto más vale un documento altamente relevante que uno marginal es "inherentemente bastante arbitrario". Los autores defienden hacer *sensitivity testing* con varias cuantificaciones (planas y empinadas) en vez de un solo esquema. Citan a Voorhees (2001), que pesó documentos altamente relevantes por factores 1 a 1000. Mencionan que Tang et al. (1999) propusieron siete como número óptimo de niveles de relevancia, aunque las medidas no están atadas a ningún número fijo.
- **Arbitrariedad de la base del descuento.** No hay una forma "específica" privilegiada de descuento; la base debe venir del escenario de evaluación. Bases límite: b→1 hace el descuento demasiado agresivo (solo importa el primer documento); b→∞ hace que DCG→CG. Recomiendan el rango 2 a 10.
- **Combinaciones de parámetros sin guía interna.** "La matemática funciona para cualquier combinación de parámetros y no puede aconsejarnos sobre cuál elegir" — la elección (último rango considerado, pesos, descuento) debe provenir del escenario de uso, no de la métrica.
- **Sin efectos de orden ni solapamiento/redundancia.** Las medidas no manejan redundancia entre documentos (instance recall del interactive track lo atiende parcialmente). La relevancia se trata como unidimensional cuando en realidad es multidimensional (Vakkari & Hakala 2000).
- **Juicios estáticos.** Ninguna medida basada en juicios de relevancia estáticos maneja cambios dinámicos en los criterios de relevancia del usuario.
- **Muestra pequeña.** "El número de tópicos (20) es bastante pequeño para dar resultados confiables", aunque sirve para ilustrar el comportamiento de las medidas.

## Impacto: la métrica de facto del ranking

Aunque el paper presenta CG, DCG y nDCG como una familia, fue **nDCG** la que se convirtió en el estándar de evaluación de ranking en las dos décadas siguientes. Su normalización a [0, 1] la hace comparable entre consultas con bases de relevancia de tamaños distintos, y su descuento logarítmico captura la intuición central de cualquier interfaz ordenada (búsqueda web, recomendación, QA): **lo que está arriba importa más**. nDCG@k es hoy la métrica reportada por defecto en *learning-to-rank*, motores de búsqueda web, sistemas de recomendación y *retrieval* para RAG. La variante moderna más común usa la ganancia exponencial $(2^{rel_i}-1)$ en el numerador —que enfatiza aún más los documentos altamente relevantes— en lugar de la ganancia lineal $rel_i$ del paper original; ambas comparten la misma estructura de descuento logarítmico e idéntica idea de normalización por el ideal.

## Conexión con la Clase 25 (Ranking Metrics)

La Clase 25 (sistemas recomendadores multimodales) cierra con la sección **Metrics** y presenta nDCG de forma explícita. El laboratorio de la clase usa la formulación moderna:

$$
nDCG_p = \frac{\sum_{i=1}^{p} \frac{2^{rel_i}-1}{\log_2(i+1)}}{\sum_{i=1}^{REL_p} \frac{2^{rel_i}-1}{\log_2(i+1)}} = \frac{DCG_p}{IDCG_p}
$$

Este es exactamente el descendiente directo del paper de Järvelin y Kekäläinen: el numerador es DCG (ganancia descontada por el log de la posición), el denominador es iDCG (el mismo cómputo sobre el ranking ideal), y el cociente es nDCG en [0, 1]. El descuento $\log_2(i+1)$ es la elección de base 2 del paper (usuario impaciente); la ganancia $2^{rel_i}-1$ es la variante exponencial que se popularizó después de 2002.

**Ejemplo numérico de la clase.** La slide de Ranking Metrics reproduce un cálculo sobre un ranking del tipo [rel, no, rel, no, …] (relevantes dispersos en posiciones bajas) con cinco documentos relevantes:

- **DCG ≈ 1.4485** — ganancia descontada del ranking real, con los relevantes esparcidos lejos del tope.
- **iDCG ≈ 2.9485** — ganancia descontada del ranking ideal. Este valor es **exactamente verificable**: con cinco documentos relevantes binarios colocados en las cinco primeras posiciones (el ideal), usando el descuento $1/\log_2(1+i)$,
$$
\mathrm{iDCG} = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} = 1 + 0.6309 + 0.5 + 0.4307 + 0.3869 \approx 2.9485.
$$
- **nDCG = DCG/iDCG ≈ 1.4485 / 2.9485 ≈ 0.4912** — el ranking real captura cerca del **49%** de la ganancia que daría el orden perfecto.

La lectura pedagógica: un nDCG de ~0.49 dice que el sistema acertó con los documentos relevantes pero los puso demasiado abajo en la lista; el ideal pondría esos mismos cinco documentos en el tope (nDCG = 1). Es exactamente la "distancia a la curva ideal" que las Figuras 3 y 4 del paper original visualizan, condensada en un solo número por consulta.

## Notas y enlaces

- **Venue:** ACM TOIS, Vol. 20, No. 4, October 2002, pp. 422–446.
- **Sin arXiv** (publicación de revista ACM previa a la era de preprints en el área).
- Antecedente directo de los propios autores: Järvelin & Kekäläinen (2000), donde CG y DCG aparecieron por primera vez; aplicado luego en el TREC Web Track 2001 (Voorhees) y en summarization (Sakai & Sparck-Jones 2001).
- Referencias clave del marco conceptual: Robertson & Belkin (1978) sobre probabilidad de relevancia; Sormunen (2002) por los juicios graduados; Korfhage (1997) por sliding ratio y satisfaction measure.
