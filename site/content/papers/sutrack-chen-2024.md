---
title: "SUTrack: seguimiento unificado de un objeto (2024)"
weight: 456
math: true
---

{{< paper-card
    title="SUTrack: Towards Simple and Unified Single Object Tracking"
    authors="Xin Chen, Ben Kang, Wanting Geng, Jiawen Zhu, Yi Liu, Dong Wang, Huchuan Lu (Dalian University of Technology / Baidu)"
    year="2024"
    venue="AAAI 2025 / arXiv:2412.19138"
    arxiv="2412.19138"
    pdf="/papers/sutrack-chen-2024.pdf" >}}
Cinco tareas de seguimiento que hasta ahora tenían cada una su arquitectura, su entrenamiento y su comunidad —RGB, RGB-Profundidad, RGB-Térmico, RGB-Eventos y RGB-Lenguaje— consolidadas en **un solo Transformer entrenado en una sola sesión**. La clave es una **representación de entrada unificada**: todas las modalidades se convierten a un formato común y se distinguen mediante un *soft token type embedding* en vez de por ramas dedicadas. El resultado supera a los modelos especializados en 11 datasets, lo que sugiere **sinergia entre modalidades**: entrenar con datos térmicos mejora el seguimiento RGB. Se ofrecen variantes desde dispositivos *edge* hasta GPU de alta gama.
{{< /paper-card >}}

---

## Qué unifica, y qué no

{{< concept-alert type="advertencia" >}}
**SUTrack es SOT, no MOT.** *Single Object Tracking*: se recibe una caja en el primer frame y hay que re-localizar ese objeto en los siguientes. No hay detección de objetos nuevos, no hay identidades múltiples que gestionar, no hay problema de asignación.

Es una rama distinta de la del [SORT](/papers/sort-bewley-2016)/[DeepSORT](/papers/deepsort-wojke-2017) que ocupa la mayor parte de la [Clase 42](/clases/clase-42). Los dos se llaman "tracking" y comparten intuiciones, pero sus benchmarks (LaSOT, GOT-10k contra MOTChallenge), sus métricas (AUC de solapamiento contra HOTA) y sus algoritmos no se solapan. Lo que SUTrack unifica son **modalidades de entrada**, no los dos paradigmas de seguimiento.
{{< /concept-alert >}}

## El problema de la fragmentación

Cada tarea multimodal de SOT desarrolló su propia solución: una arquitectura con ramas específicas para profundidad, otra para térmico, otra para eventos. Las consecuencias que el paper señala son procesos de entrenamiento redundantes, innovaciones técnicas repetidas en cada rama, y **conocimiento cross-modal que no se comparte**.

La propuesta: si todas las modalidades auxiliares (profundidad, térmico, eventos) pueden expresarse como imágenes, y el lenguaje como tokens de un encoder de texto, entonces una sola arquitectura las procesa todas.

## Cómo funciona

**Representación unificada.** La imagen RGB y la modalidad auxiliar se concatenan en el canal y se dividen en parches de $P \times P \times 6$. Para el RGB puro, la modalidad auxiliar se rellena; para RGB-Lenguaje, el texto pasa por CLIP-L y se inyecta como tokens adicionales. Un único encoder Transformer procesa la secuencia.

**Soft token type embedding.** En vez de un embedding de tipo discreto por modalidad —que forzaría a decidir a qué categoría pertenece cada entrada—, se aprende una combinación **suave** de embeddings de tipo. Esto permite que modalidades parecidas compartan representación y que el modelo sea, en la práctica, agnóstico a la modalidad.

**Entrenamiento con reconocimiento de tarea auxiliar.** Una pérdida adicional que predice de qué tarea proviene cada muestra, usada solo durante el entrenamiento, mejora el desempeño con costo despreciable.

## Resultados

SUTrack-B384 alcanza **74,4 % de AUC en LaSOT** y establece nuevos máximos en varios de los 11 datasets evaluados, superando a los métodos específicos de cada tarea. La variante T224 corre en tiempo real en dispositivos *edge*; las mayores requieren GPU.

El hallazgo interesante es el de la **sinergia**: el modelo unificado supera a los especialistas, lo que implica que los datos de una modalidad ayudan a las otras. Es el mismo patrón que se observó en NLP multilingüe y en modelos multitarea de visión — y el argumento estándar a favor de la unificación.

## Por qué importa para la Clase 42

Es el primero de los "modelos integrados" con que cierra la [Clase 42](/clases/clase-42), y el punto donde la clase salta de 2017 a 2024. El contraste con SORT es la lección: **de un algoritmo de cien líneas sin parámetros aprendidos a un Transformer multimodal entrenado sobre once datasets**.

Conviene tener presente, al leer esa transición, que no es una línea recta de progreso sobre la misma tarea. SORT y SUTrack resuelven problemas distintos, y en MOT el estado del arte de 2022-2023 seguía siendo Kalman más húngaro con correcciones ([OC-SORT](/papers/oc-sort-cao-2022), [ByteTrack](/papers/bytetrack-zhang-2021)). La unificación multimodal es una tendencia real; no reemplazó al tracking-by-detection.

---

**Ver también:** [SAM 3 (2025)](/papers/sam3-meta-2025) · [SORT (2016)](/papers/sort-bewley-2016) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos) · [Vision Transformer](/fundamentos/vision-transformer) · [Foundation Models](/fundamentos/foundation-models)
