---
title: "Learning to Represent Programs with Graphs (2018)"
weight: 308
math: true
---

{{< paper-card
    title="Learning to Represent Programs with Graphs"
    authors="Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi"
    year="2018"
    venue="ICLR 2018"
    pdf="/papers/programs-as-graphs-allamanis-2018.pdf"
    arxiv="1711.00740" >}}
Paper fundacional de la línea **GNN para código fuente**. Su tesis: un programa no es solo una secuencia de tokens ni solo un árbol de sintaxis (AST), sino un **grafo** que combina el AST con aristas semánticas de flujo de datos y de control que un compilador ya sabe calcular. Sobre ese grafo entrenan una [Gated Graph Neural Network](/papers/ggnn-li-2015) (GGNN) para dos tareas: **VarNaming** (predecir el nombre de una variable) y **VarMisuse** (detectar cuándo se usó la variable equivocada, un bug). Validado sobre 2.9 millones de líneas de C# de GitHub, VarMisuse alcanza 85.5% de accuracy y **encontró bugs reales** en proyectos maduros como Roslyn y RavenDB. Nota para la [Clase 27](/clases/clase-27): el primer autor, Miltos Allamanis, es además autor del tutorial de GNN que la clase cita como material base.
{{< /paper-card >}}

---

## Contexto: el código no es solo texto

Hacia 2017, cuando se aplicaba aprendizaje automático a código fuente, casi todo el mundo trataba el programa como una **secuencia de tokens** (modelos de lenguaje al estilo NLP) o, a lo más, como un **árbol de sintaxis** (AST). El fenómeno del *big code* —repositorios masivos de código en GitHub más métodos de ML escalables— prometía sistemas que aprendieran de código existente para asistir a los ingenieros. El problema era *cómo* darle el código al modelo.

Ambas representaciones desperdician la información más valiosa que tiene el código y que el lenguaje natural no tiene: una **semántica conocida y bien definida**, extraíble con herramientas de compilador estándar. El mismo nombre de variable reutilizado en dos puntos distantes del archivo, el flujo de datos desde una asignación hasta su uso, la relación entre un argumento y su parámetro formal: todo eso son **dependencias de largo alcance** que una secuencia lineal captura mal y que un AST puro tampoco modela. Y son exactamente las dependencias que un ingeniero usa para razonar sobre el código.

La propuesta central: representar cada programa como un **grafo** cuyo esqueleto es el AST pero que se enriquece con **aristas semánticas** adicionales (flujo de datos, flujo de control, alcances léxicos, tipos), y entrenar una GGNN sobre ese grafo. La tesis de fondo —y la lección transversal de la clase— es que *exponer la estructura explícitamente como entrada estructurada reduce las exigencias sobre cantidad de datos, capacidad del modelo y régimen de entrenamiento*, permitiendo resolver tareas fuera del estado del arte de la época.

Las tareas se enmarcan como un problema de "rellenar el hueco" (*fill the blank*), emparentado con el aprendizaje de representaciones distribuidas de palabras tipo Word2Vec. La diferencia es que aquí se aprende de una estructura mucho más rica que la mera coocurrencia textual.

## Representar programas como grafos

### El esqueleto: el AST

El grafo parte del **AST** del programa, con dos tipos de nodos:

- **Nodos de sintaxis** (*syntax nodes*): los **no-terminales** de la gramática (`ExpressionStatement`, `InvocationExpression`, `ArgumentList`), etiquetados con el nombre del no-terminal.
- **Tokens de sintaxis** (*syntax tokens*): los **terminales** (`Assert`, `.`, `NotNull`, `(`), etiquetados con el string que representan.

### Las aristas: aquí está la riqueza

El paper define **10 tipos de aristas** que, con sus inversas, dan 20 tipos en el GGNN. Esta es la pieza central de la contribución:

**Aristas sintácticas** (el esqueleto del AST):
- **Child:** conecta padre con hijo según el AST.
- **NextToken:** conecta cada token con su sucesor textual (las aristas Child no inducen orden entre hermanos).

**Aristas semánticas de flujo de datos**, calculables con análisis de compilador:
- **LastRead / LastUse:** los tokens donde la variable pudo usarse por última vez (puede haber varios tras un condicional, e incluso tokens *posteriores* en loops).
- **LastWrite:** los tokens donde la variable se escribió por última vez.
- **ComputedFrom:** ante `v = expr`, conecta `v` con todas las variables que aparecen en `expr`.

**Aristas semánticas adicionales:**
- **LastLexicalUse:** encadena todos los usos léxicos de una misma variable, ignorando el flujo de datos (enlaza la `v` de la rama `then` con la del `else`).
- **ReturnsTo:** conecta cada `return` con la declaración del método (atajo hacia su nombre y tipo).
- **FormalArgName:** conecta argumentos de una llamada con sus parámetros formales (`Foo(bar)` con `Foo(InputStream stream)` enlaza `bar` con `stream`).
- **GuardedBy / GuardedByNegation:** conecta cada variable con las expresiones de guarda (condicionales) que la protegen.

Por cada tipo se introduce su **arista inversa** (transponiendo la adyacencia): ayudan a propagar información más rápido por la GGNN y hacen el modelo más expresivo.

### Información de tipos

El método asume un **lenguaje estáticamente tipado y compilable** (C#), de modo que cada variable tiene un tipo conocido. Para aprovechar la **jerarquía de tipos** orientada a objetos, cada variable se mapea al conjunto de sus supertipos y su representación es el **máximo elemento a elemento** sobre los embeddings de ese conjunto (pooling natural para los *type lattices*, que son órdenes parciales). Usar todos los supertipos permite **generalizar a tipos no vistos** que implementan interfaces comunes (`List<int>` y `List<string>` comparten `IList`). La representación inicial de cada nodo combina ese embedding de tipo con el nombre del token, partido en *subtokens* en camelCase y pascal_case (`classTypes` → `class` + `types`).

## El motor: Gated Graph Neural Networks

El modelo se construye sobre [GGNN](/papers/ggnn-li-2015) (Li, Tarlow, Brockschmidt y Zemel, 2015; nótese que Brockschmidt es coautor de ambos papers). El esquema es [message passing](/fundamentos/message-passing) clásico: cada nodo $v$ mantiene un **vector de estado** $h^{(v)}$; en cada paso envía mensajes de tipo $k$ a sus vecinos vía una capa lineal $m_k^{(v)} = f_k(h^{(v)})$; cada nodo **agrega por suma** los mensajes entrantes $\tilde{m}^{(v)}$; y actualiza su estado con una celda recurrente **GRU**: $h'^{(v)} = \text{GRU}(\tilde{m}^{(v)}, h^{(v)})$.

Esto se repite **8 pasos de propagación** (menos era insuficiente, más no ayudaba) y los estados finales son las representaciones de nodo. Los autores notan que las *Graph Convolutional Networks* (GCN) serían un reemplazo más simple, pero en sus experimentos **generalizaron peor** que las GGNN.

## Las dos tareas

### VarNaming: predecir el nombre de una variable

Dada una variable, se reemplaza su nombre en *todos* sus tokens por un `<SLOT>` especial, se corre la GGNN y se promedian los estados de los slots para obtener una representación de uso. Esa representación inicializa una GRU que predice el nombre como **secuencia de subtokens** (`inputStreamBuffer` → `[input, stream, buffer]`). Es una arquitectura **graph2seq** entrenada con máxima verosimilitud.

### VarMisuse: detectar la variable equivocada

Es la contribución conceptual más original. El caso paradigmático (un bug real en RavenDB): el código declara `clazz` y `first`, ambos del mismo tipo, y llama `Assert.NotNull(clazz)` donde debería decir `Assert.NotNull(first)`. Como ambas son del mismo tipo, **el compilador no emite ningún error**: es válido sintáctica y tipológicamente, pero semánticamente está mal (típico fruto de copiar-pegar). Un ingeniero lo detectaría; el análisis estático clásico no.

**Formalización.** Un token de variable cuyo valor correcto queremos predecir es un **slot**. Dado el contexto, hay que elegir la variable correcta de entre todas las que son **type-correct y en scope** en esa posición. En promedio hay **3.8 candidatas type-correct por slot**, así que no basta con filtrar por tipo: hay que razonar sobre el rol semántico. En el grafo se inserta un nodo `<SLOT>` conectado con las aristas que *no* dependen de la variable elegida, y un nodo "candidato" por cada variable posible con las aristas que tendría *si estuviera en el slot*. La variable correcta es $\arg\max_v W[c(t), u(t,v)]$ sobre las representaciones de contexto y uso, entrenado con un objetivo **max-margin**.

## Experimentos

**Dataset.** Corpus de **29 proyectos C# open source de GitHub** (los no-fork con más estrellas que compilaban completamente con Roslyn, necesario para extraer tipos precisos), con cerca de **2.9 millones de líneas** de código: Roslyn, RavenDB, EntityFramework, Newtonsoft.Json, Akka.NET, entre otros. Crucialmente, se separó un **UnseenProjTest** (proyectos completamente desconocidos) del **SeenProjTest** (split por archivos dentro de proyectos vistos), para medir generalización a dominios nuevos.

**Resultados.** La GGNN supera a los baselines secuenciales (RNN bidireccionales) por amplio margen, sobre todo en VarMisuse, donde la estructura importa más:

| Tarea | Métrica | Baseline biRNN | **GGNN** | GGNN (Unseen) |
|---|---|---|---|---|
| VarMisuse | Accuracy | 73.7 | **85.5** | 78.2 |
| VarMisuse | PR AUC | 0.941 | **0.980** | 0.958 |
| VarNaming | Accuracy | 42.9 | **53.6** | 44.0 |
| VarNaming | F1 | 50.1 | **65.8** | 62.0 |

En UnseenProjTest el rendimiento baja —el type lattice y el vocabulario del proyecto son mayormente desconocidos— pero sigue siendo bueno. El **ablation** es revelador: restringirse a **solo aristas sintácticas** (el AST puro, que era el estado del arte previo) hunde VarMisuse de 85.5% a **55.3%**, confirmando que el flujo de datos y las aristas semánticas son lo que hace funcionar la tarea.

**Bugs reales.** La validación más contundente: revisando manualmente las top-500 ubicaciones donde el modelo confiaba más en una variable distinta de la original, los autores **encontraron 3 bugs en RavenDB y 3 en Roslyn** (el propio compilador de C# de Microsoft). Reportaron los de Roslyn a los desarrolladores, que los corrigieron; uno podía causar un crash de Visual Studio. A un 10% de falsos positivos (límite aceptado en industria), el modelo logra 73% de true positive rate. La conclusión práctica: el modelo puede guiar la revisión de código o priorizar dónde concentrar testing y análisis caro.

## Limitaciones

- **Requiere compilación completa y tipado estático.** Toda la riqueza semántica depende de compilar el proyecto con Roslyn. Esto excluye lenguajes dinámicos, código incompleto o proyectos que no compilan (los autores descartaron proyectos por esta razón).
- **Generalización limitada a proyectos nuevos:** el rendimiento cae en UnseenProjTest porque el type lattice y el vocabulario son específicos del proyecto.
- **Aliases:** cuando dos variables apuntan al mismo valor en memoria, cualquier elección es correcta en la práctica, pero el modelo es penalizado si no coincide con la original.
- **Razonamiento interprocedural:** falla cuando entender el slot exige razonar cruzando fronteras de métodos o clases.
- **Costo de construcción del grafo:** requiere herramientas de análisis de compilador y preprocesamiento intensivo; no es plug-and-play sobre texto crudo.

## Impacto

Este es uno de los trabajos fundacionales de la línea **GNN para código fuente** y, más ampliamente, de *neural program analysis*. Estableció que el código se modela mejor como grafo que como secuencia, idea que permeó la siguiente generación de modelos de código. Es antecedente directo de:

- **Análisis estático neuronal y detección de bugs por aprendizaje:** un modelo aprendido puede complementar (no reemplazar) las herramientas clásicas, señalando ubicaciones "inusuales" para revisión humana.
- **Asistentes de código y copilots:** aunque los copilots modernos basados en LLMs volvieron al paradigma de secuencia de tokens (por escalabilidad y porque los Transformers grandes aprenden estructura implícitamente), la tarea VarMisuse y la representación de grafo siguieron vivas en sistemas de detección de bugs y *code review* automatizados, y en trabajos que combinan grafos con Transformers (GREAT, GraphCodeBERT).

El dataset y la formulación de VarMisuse se volvieron benchmark de referencia para evaluar modelos de comprensión de código.

## Por qué importa para la Clase 27

La [Clase 27](/clases/clase-27) sobre [redes neuronales de grafos](/fundamentos/redes-neuronales-de-grafos) usa este paper como cierre casi perfecto, por cuatro razones entrelazadas:

1. **Es la aplicación "Bugs in Code" del temario.** La clase presenta la detección de bugs como caso de uso de GNN; este paper *es* ese caso, con bugs reales encontrados en Roslyn y RavenDB.
2. **Usa GGNN, el modelo que la clase explicó.** Muestra la Gated Graph Neural Network y su [message passing](/fundamentos/message-passing) trabajando a escala real sobre grafos de miles de nodos, con la ingeniería de producción que hace falta (tensores sparse, batch como un grafo gigante).
3. **Es la tesis transversal del curso hecha concreta:** cualquier dato con estructura relacional es un grafo. Si incluso el código fuente —lo más "secuencial" del mundo— se modela mejor como grafo, el mensaje es claro para los [datos estructurados](/dominios/estructurados): tablas con foreign keys, registros de pacientes vinculados, moléculas, redes sociales. Conecta de forma natural con *record linkage* y emparejamiento de entidades (como FHIR patient matching), donde las relaciones entre registros son tan informativas como los atributos individuales.
4. **El autor es el del tutorial base de la clase.** Miltos Allamanis, primer autor, escribió *"An Introduction to Graph Neural Networks: Models and Applications"*, el tutorial que la clase cita en sus créditos. Estudiar este paper es ver al autor del material didáctico aplicando, en un trabajo de investigación real y de alto impacto, exactamente los conceptos que enseña.

La lección final: *exponer estructura explícitamente reduce la necesidad de datos y de capacidad del modelo*. Antes de escalar a un modelo gigante, vale la pena preguntarse qué estructura del problema podemos darle "gratis" al modelo, porque muchas veces ya está ahí, esperando ser modelada como un grafo.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1711.00740 (v3, mayo 2018)
- Código de referencia GGNN: https://github.com/Microsoft/gated-graph-neural-network-samples
- Dataset: https://aka.ms/iclr18-prog-graphs-dataset
- Afiliaciones: Microsoft Research (Cambridge, UK) y Simon Fraser University.
