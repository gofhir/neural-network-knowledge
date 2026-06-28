---
title: "Actividades (1-6) resueltas"
weight: 5
---

Las seis actividades del práctico, resueltas y justificadas. Todas fueron verificadas contra el [notebook ejecutado](/notebooks/lab27.ipynb). Lo importante no es solo marcar la opción correcta, sino **entender por qué las demás están mal** — eso fundamenta la respuesta.

## Actividad 1 — el bias inductivo

> *Al aplicar una GCN sobre Karate Club se aprecia un ordenamiento de los nodos por comunidad. ¿De dónde proviene esa información, considerando que los pesos están aún en sus valores aleatorios iniciales?*

Hay **tres fuentes posibles** de información en cualquier GNN; auditémoslas:

| Fuente | ¿Aporta señal de comunidad? |
|---|---|
| Features `x` | ❌ Son **one-hot** (identidad 34×34): ortogonales, sin noción de grupo |
| Pesos `W` | ❌ Están **aleatorios sin entrenar** (la premisa) |
| Estructura (aristas) vía la operatoria GCN | ✅ Única fuente posible |

Aunque $\mathbf{W}$ sea aleatorio, la suma $\sum_{j \in \mathcal{N}(i)}$ **promedia los estados de los vecinos** siguiendo las **aristas**. Dos nodos que comparten vecinos reciben entradas parecidas y, tras la misma $\mathbf{W}$ compartida, salen con embeddings parecidos. Apilar capas propaga el efecto. Es el *inductive bias* de homofilia.

**Respuesta:** `Provienen de la operatoria de la GCN que obedece a las aristas del Grafo`

> El distractor más fino es "...obedece a los **features** de los Nodos": atribuye la señal a los features, que aquí son inútiles. La clave es que la operatoria sigue las **aristas**.

## Actividad 2 — nodos sin aristas entrantes

> *Ocurre un problema cuando hay nodos con solo aristas salientes (ninguna entrante). Describa el problema y una solución.*

El escenario solo tiene sentido en un grafo **dirigido** (ej. citas: un paper nuevo que cita a otros pero que nadie citó todavía). Un nodo actualiza su estado agregando lo que **le llega** —sus vecinos entrantes—:

$$
\mathbf{h}_i^{(t+1)} = \mathbf{W}^{(t+1)} \sum_{j \in \mathcal{N}_{\text{in}}(i)} \tfrac{1}{c_{i,j}}\,\mathbf{h}_j^{(t)}
$$

Si $\mathcal{N}_{\text{in}}(i) = \varnothing$, la suma es **vacía = 0**: el nodo se "borra", pierde su propia información y queda en un estado nulo/constante, inútil para clasificar.

**La solución (truco de la clase): self-loops** ($\tilde{A} = A + I$). Conectar cada nodo consigo mismo garantiza que **siempre** tenga una arista entrante (la suya), preservando su información. Es lo que `GCNConv` hace por defecto.

```python
Problema = 'Un nodo sin aristas entrantes no recibe mensajes: su suma de agregación queda vacía, su embedding se vuelve cero/constante y pierde su propia información, quedando inservible para clasificar.'
Una_posible_solucion = 'Agregar self-loops (A+I), conectando cada nodo consigo mismo para que siempre tenga una arista entrante propia y preserve su información; es lo que GCNConv hace por defecto.'
```

## Actividad 3 — nodo completamente aislado

> *Un nodo está totalmente aislado (sin entrantes ni salientes). (3.1) ¿Sirve la solución anterior? (3.2) Sin aplicarla, tras una GCN de 3 capas, ¿cómo será su estado final?*

**3.1 → Sí.** El self-loop no lo conecta con otros (es imposible, no tiene vecinos), pero sí le permite **preservar y transformar su propia información**: su vecindario pasa de $\varnothing$ a $\{i\}$, así procesa su feature en cada capa en vez de borrarse. Eso permite clasificarlo por su propio contenido.

**3.2 → Puede ser distinto al inicial.** La intuición ingenua ("no habla con nadie → no cambia") es falsa. Sin self-loop, la suma es vacía y ya en la **primera capa** $\mathbf{h}_i^{(1)} = \mathbf{W}\cdot\mathbf{0} = \mathbf{0}$ (o el bias). El feature inicial $\mathbf{x}_i$ **nunca entra** en la fórmula → se pierde. El estado final es radicalmente distinto: cero/constante.

```python
Sirve_la_solucion_anterior = 'Si'
Como_es_su_estado_final = 'Puede ser distinto al inicial'
```

| Caso | Sin self-loop | Con self-loop |
|---|---|---|
| **A2:** solo salientes | se borra | preserva su info |
| **A3:** aislado | se borra a cero/bias | preserva y transforma su info |

## Actividad 4 — grado promedio en una línea

> *Escriba código que imprima el grado promedio de un grafo `Data`. (Hint: PyG guarda las aristas no dirigidas dos veces.)*

Como cada arista no dirigida está duplicada en COO, la suma de grados de todos los nodos es exactamente `num_edges`. Por eso el grado promedio se obtiene **sin dividir por 2**:

```python
Codigo = "print(f'Grado promedio: {data.num_edges / data.num_nodes:.2f}')"
```

Verificación: Karate Club da `156 / 34 ≈ 4.59`. El ejemplo del enunciado (10 nodos todos-con-todos): `num_edges = 90`, `90/10 = 9` ✓. La respuesta ya aparece literalmente en cada bloque diagnóstico del notebook (`Average node degree: ...`).

## Actividad 5 — dimensiones de `GCNConv(3, 2)`

> *En el modelo dado: ¿qué representan el 3 y el 2 en `GCNConv(3, 2)`? ¿Se puede saber el número de clases?*

`GCNConv(in_channels, out_channels)`, igual que `nn.Linear`. El flujo dimensional `... → conv2(4,3) → conv3(3,2) → classifier(2,4)` confirma que el `3` debe ser la salida de `conv2`.

- **El 3** = features de **entrada** de `conv3` (coincide con la salida de `conv2`).
- **El 2** = features de **salida** de `conv3` (embedding de 2D por nodo).
- **Clases → Sí, son 4.** No lo dicen las `GCNConv` (son hidden, arbitrarias), sino la **última capa**: `classifier = Linear(2, 4)` produce 4 logits, y al entrenar con CrossEntropy la salida del clasificador **es** el número de clases.

```python
Lo_que_representa_el_3 = 'La dimensión de features de ENTRADA de la capa conv3 (coincide con la salida de conv2, que es 3).'
Lo_que_representa_el_2 = 'La dimensión de features de SALIDA de conv3: el embedding de 2 dimensiones que produce por nodo.'
Es_posible_conocer_el_numero_de_clases = 'Si'
Justificacion_o_numero_de_clases = 'Son 4 clases: la capa final classifier = Linear(2, 4) produce 4 logits, y al entrenar con CrossEntropy la salida del clasificador debe igualar el número de clases.'
```

> Distractor: el `2` de `conv3` **no** son "2 clases" — es la dimensión del embedding intermedio. Las clases las define solo la salida del `classifier`.

## Actividad 6 — campo receptivo y distancia

> *Mismo modelo (3 capas). Los nodos A y B están a 4 aristas. ¿Pudo la clasificación de A considerar información de B?*

Cada capa GCN propaga información **un salto**: tras $k$ capas, un nodo "ve" hasta $k$ saltos (su **campo receptivo**). El modelo tiene **3 capas `GCNConv`** (el `classifier = Linear` no propaga: opera por nodo, sin `edge_index`). Alcance = **3 saltos**.

B está a **4 aristas** → fuera del radio de 3. Su información **nunca llega** a A. Harían falta ≥4 capas.

```python
Es_posible_o_no = 'No'
Justificacion = 'El modelo tiene 3 capas GCNConv y cada capa propaga información solo 1 salto, por lo que el campo receptivo de A es de 3 aristas. B está a 4 aristas (fuera de ese radio), así que su información nunca llega a A. El classifier final es lineal por nodo y no propaga. Harían falta al menos 4 capas GCN para que A considerara a B.'
```

{{< callout type="info" >}}
**Alcance vs discriminabilidad.** Más capas ⇒ más alcance, pero **no se puede apilar indefinidamente**: aparece el **over-smoothing** (con muchas capas todos los embeddings convergen al mismo valor y los nodos se vuelven indistinguibles). En *record linkage*, este "radio de $k$ saltos" es cuánta vecindad relacional incorporás al decidir si dos registros son la misma persona: si la evidencia conectora está a más saltos que capas tengas, el modelo no la usa.
{{< /callout >}}

## Resumen

| # | Respuesta |
|---|---|
| **1** | Provienen de la operatoria de la GCN que obedece a las aristas del Grafo |
| **2** | Nodo sin entrantes → suma vacía → se borra. Solución: self-loops (A+I) |
| **3** | Sirve: **Sí** · Estado final: **Puede ser distinto al inicial** |
| **4** | `print(data.num_edges / data.num_nodes)` |
| **5** | 3 = entrada de conv3 · 2 = salida de conv3 · Clases: **Sí, 4** |
| **6** | **No** — 3 capas alcanzan 3 saltos; B está a 4 |
