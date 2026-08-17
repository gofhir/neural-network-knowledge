---
title: "SAM 3: Segment Anything with Concepts (2025)"
weight: 457
math: true
---

{{< paper-card
    title="SAM 3: Segment Anything with Concepts"
    authors="Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu et al. (Meta Superintelligence Labs)"
    year="2025"
    venue="ICLR 2026 (en revisión) / arXiv:2511.16719"
    arxiv="2511.16719"
    pdf="/papers/sam3-meta-2025.pdf" >}}
La generalización del *prompting* al seguimiento: en vez de dar una caja inicial, se da un **concepto** —una frase nominal corta como *"autobús escolar amarillo"*, un ejemplar de imagen, o ambos— y el modelo detecta, segmenta y **sigue con identidades persistentes** todas las instancias que coinciden, en imágenes y en video. Meta llama a la tarea *Promptable Concept Segmentation* (PCS). La contribución arquitectónica destacable es la **cabeza de presencia**, que separa el "¿está presente este concepto?" del "¿dónde está?", y el desacople explícito entre un detector agnóstico a la identidad y un rastreador cuya única función es separar identidades. Alcanza **48,8 de mask AP zero-shot en LVIS** contra 38,5 del mejor modelo previo.
{{< /paper-card >}}

---

## La tarea

PCS toma un *prompt* conceptual y devuelve máscaras de instancia **para todos los objetos que coinciden**, preservando identidades a lo largo del video. La diferencia con SAM y SAM 2 es cuantitativa en la formulación y cualitativa en el uso: las versiones anteriores segmentaban **un** objeto por *prompt* (un clic, una caja); SAM 3 segmenta **todas las apariciones** de un concepto.

El texto se restringe deliberadamente a **frases nominales simples** — el modelo no está diseñado para expresiones referenciales largas ni para consultas que requieran razonamiento. Para eso, los autores muestran que se puede componer con un LLM multimodal que traduzca la consulta compleja a frases nominales.

## Arquitectura

Un **encoder visual compartido**, y sobre él dos módulos deliberadamente separados:

- **Detector** basado en DETR, condicionado por texto, geometría y ejemplares de imagen.
- **Rastreador** que hereda la arquitectura encoder-decoder con memoria de SAM 2, y soporta refinamiento interactivo.

{{< concept-alert type="clave" >}}
**Por qué separarlos.** El paper da la razón explícita: *el detector necesita ser agnóstico a la identidad, mientras que el objetivo principal del rastreador es separar identidades en el video*. Son objetivos en conflicto sobre la misma representación, y el desacople evita el choque.

Es exactamente el mismo diagnóstico de [FairMOT](/papers/fairmot-zhang-2020) cinco años antes, sobre otra escala y otra arquitectura: cuando dos tareas comparten backbone, hay que decidir explícitamente qué se comparte y qué no. FairMOT lo resolvió equilibrando; SAM 3 lo resuelve separando.
{{< /concept-alert >}}

**La cabeza de presencia.** La novedad técnica destacada. En detección de vocabulario abierto, cada consulta tiene que responder dos preguntas a la vez —¿existe este concepto en la imagen? ¿dónde está?— y eso degrada ambas. SAM 3 introduce un **token global de presencia** que responde solo la primera, y deja a las consultas la localización. Las ablaciones confirman que aporta, y que su efecto es mayor cuando el entrenamiento incluye **negativos difíciles** — frases que describen conceptos ausentes pero plausibles.

## El motor de datos

Buena parte del resultado viene de los datos, no de la arquitectura. El pipeline usa LLMs multimodales como **anotadores** (generan frases nominales y negativos difíciles a partir de una ontología) y como **verificadores** (afinados para alcanzar precisión cercana a la humana), reservando el trabajo humano para corregir los casos que la IA marca como erróneos. Duplica el rendimiento de anotación respecto de un pipeline solo humano.

El resultado: **4 millones de frases únicas y 52 millones de máscaras** de alta calidad, más un dataset sintético de 38 M de frases y 1,4 mil millones de máscaras. Y un benchmark nuevo, SA-Co, con 207 000 conceptos — más de 50 veces los conceptos de los benchmarks existentes.

## Resultados y costo

- **48,8 de mask AP zero-shot en LVIS**, contra 38,5 del mejor modelo previo (+10,3 puntos absolutos).
- Al menos **2×** sobre las líneas base en el benchmark SA-Co, tanto en imagen como en video.
- Mejora sobre SAM 2 en las tareas de *prompt* visual heredadas.
- **30 ms por imagen con más de 100 objetos detectados** en una GPU H200. En video la latencia escala con el número de objetos y se sostiene cerca de tiempo real para unos **5 objetos concurrentes**.

{{< concept-alert type="advertencia" >}}
Ese último dato conviene retenerlo junto al de la clase: SAM 3 mantiene tiempo real para **~5 objetos concurrentes en una H200**. [SORT](/papers/sort-bewley-2016) corre a 260 Hz con decenas de objetos en **un núcleo de CPU**. Son herramientas de mundos distintos, y la elección entre ellas casi nunca es una cuestión de exactitud sino de presupuesto de cómputo y de si el vocabulario de objetos está cerrado o abierto.
{{< /concept-alert >}}

**Sobre la cifra de la clase.** La [Clase 42](/clases/clase-42) reporta *"22 % de mejora en LVIS Zero-Shot AP vs. el mejor modelo anterior"*. La versión publicada del paper reporta 48,8 contra 38,5 — una mejora de **10,3 puntos absolutos**, equivalente a un 26,8 % relativo. El 22 % no aparece con ese valor en el texto del artículo; probablemente venga de una nota de prensa o de una métrica distinta. La cifra verificable contra el paper es la de 48,8 / 38,5.

## Por qué importa para la Clase 42

Es el último modelo que la [Clase 42](/clases/clase-42) presenta, y el que mejor ilustra hacia dónde se movió el campo: del seguimiento de **categorías fijas** (peatones, autos) al de **conceptos arbitrarios expresados en lenguaje natural**. El ejemplo de la clase —*"Track all yellow buses"*, y SAM 3 asigna IDs a cada instancia que coincide con el prompt semántico— captura exactamente el cambio de interfaz.

Lo que no cambia es la estructura del problema. SAM 3 sigue teniendo un detector y un módulo de asociación con memoria, sigue teniendo que decidir qué instancia del frame $t$ es cuál del frame $t-1$, y sigue enfrentando la oclusión. El vocabulario se abrió; el problema de la identidad sigue siendo el mismo que en 2016.

---

**Ver también:** [SUTrack (2024)](/papers/sutrack-chen-2024) · [FairMOT (2020)](/papers/fairmot-zhang-2020) · [SORT (2016)](/papers/sort-bewley-2016) · [Vision-Language Models](/fundamentos/vision-language-models) · [Foundation Models](/fundamentos/foundation-models) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
