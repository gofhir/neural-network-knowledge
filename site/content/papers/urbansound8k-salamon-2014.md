---
title: "UrbanSound8K: taxonomía y dataset de sonido urbano (2014)"
weight: 412
math: true
---

{{< paper-card
    title="A Dataset and Taxonomy for Urban Sound Research"
    authors="Justin Salamon, Christopher Jacoby, Juan Pablo Bello (NYU)"
    year="2014"
    venue="ACM Multimedia 2014"
    pdf="/papers/urbansound8k-salamon-2014.pdf" >}}
El paper identifica **dos barreras** que frenaban la clasificación automática de sonido urbano: la **falta de una taxonomía común** para nombrar las fuentes de sonido, y la **escasez de datos anotados, grandes y del mundo real**. Para atacarlas aporta tres piezas encadenadas: una **taxonomía de sonidos urbanos** jerárquica; **UrbanSound**, el dataset libre más grande de eventos sonoros urbanos etiquetados de su momento (27 horas de campo, 18.5 anotadas en 10 clases); y **UrbanSound8K**, un subconjunto de **8.732 fragmentos** de a lo más **4 segundos**, repartidos en **10 folds predefinidos** para validación cruzada reproducible. Un baseline de **MFCC + clasificadores clásicos** caracteriza las dificultades del problema. Para la [Clase 37](/clases/clase-37) es el ejemplo canónico de cómo se construye un benchmark de **clasificación de sonido ambiental (ESC)**.
{{< /paper-card >}}

---

## Contexto: por qué faltaban datos y una taxonomía común

La clasificación de sonido ambiental crecía impulsada por las redes de sensores multimedia y por el contenido urbano subido a repositorios en línea, pero era escasa comparada con áreas maduras como el habla, la música o la bioacústica. Peor aún: cuando existía trabajo urbano, solía clasificar el **tipo de escena** (calle, parque) en lugar de **identificar las fuentes de sonido** dentro de ella (bocina, motor al ralentí, canto de pájaro). El paper apunta a lo segundo, que es lo verdaderamente útil.

Diagnostican dos obstáculos concretos. El primero, **falta de datos etiquetados**: el trabajo previo se apoyaba en audio de películas o TV producidos, en entornos muy específicos, o en datasets propietarios; el enorme costo de anotar grabaciones reales mantenía esos conjuntos pequeños (el dataset del *IEEE AASP Challenge* tenía apenas 24 grabaciones por cada una de 17 clases). El segundo, **falta de un vocabulario común**: sin taxonomía compartida, cada estudio agrupaba los sonidos distinto, volviendo **imposible comparar resultados**. La tesis es que ambos problemas se resuelven juntos: la taxonomía da el marco conceptual y las etiquetas; un dataset grande, real y libre da la evidencia empírica.

## Composición: taxonomía, dataset y los 10 folds

La **taxonomía** se ancla en cuatro grupos de nivel superior comunes a la mayoría de las propuestas:

$$\text{human} \quad\bullet\quad \text{nature} \quad\bullet\quad \text{mechanical} \quad\bullet\quad \text{music}$$

Cada grupo se ramifica en subgrupos y hojas concretas (bocina de auto, martillo neumático, sirena…). Una decisión distintiva: para priorizar los sonidos relevantes, los autores examinaron **más de 370.000 quejas por ruido** al servicio 311 de Nueva York (2010 en adelante) y construyeron la taxonomía en torno a las fuentes más denunciadas —construcción, tráfico, música fuerte, aires acondicionados, ladridos.

De las hojas seleccionan **10 clases**: aire acondicionado, bocina de auto, niños jugando, ladrido de perro, perforación, motor al ralentí, disparo, martillo neumático, sirena y música callejera. El audio se recolectó desde **Freesound** (repositorio con licencia Creative Commons): descargaron más de 3.000 grabaciones (~60 horas) por consulta de clase, y tras un **filtrado manual** que conservó solo grabaciones de campo reales quedaron 1.302 grabaciones (~27 horas). Sobre ellas, con Audacity, etiquetaron los **tiempos de inicio y fin de cada ocurrencia** más una descripción de **saliencia** —primer plano (FG) o fondo (BG)—, produciendo 3.075 ocurrencias (18.5 horas). Esa colección completa es **UrbanSound**.

**UrbanSound8K** deriva de allí fragmentos cortos. El límite de 4 s viene de un trabajo previo (Chu et al.) que halló que 4 segundos bastan para que humanos identifiquen sonidos ambientales con 82% de exactitud; las ocurrencias más largas se segmentan con ventana deslizante de *hop* 2 s, y un tope de 1.000 fragmentos por clase da **8.732 fragmentos (8.75 horas)**.

La pieza que lo convierte en benchmark es la **división en 10 folds**. Como muchos fragmentos provienen de una misma grabación original de Freesound, un split completamente aleatorio podría dejar fragmentos de la *misma grabación* en train y test a la vez —una **fuga de información** que infla la exactitud, porque el clasificador reconoce la grabación en vez de generalizar. Por eso el reparto respeta dos restricciones:

$$\textbf{(R1) } \text{todos los fragmentos de una misma grabación} \rightarrow \text{el mismo fold}$$
$$\textbf{(R2) } \text{balancear el número de fragmentos por fold para cada clase}$$

De aquí la regla de oro operativa: **no re-barajar los datos**. Rehacer los folds aleatoriamente rompe R1, reintroduce la fuga por grabación y produce cifras optimistas incomparables con la literatura.

## Impacto

UrbanSound8K se convirtió en el **benchmark estándar de facto para clasificación de sonido ambiental urbano**. Su combinación de tamaño razonable, licencia libre, 10 clases bien definidas y —crucialmente— **folds predefinidos** lo transformó en el conjunto de referencia sobre el que se midieron durante años los avances del área, sobre todo cuando el deep learning llegó al audio. El propio Salamon lo usó como base de trabajos muy influyentes sobre CNN para ESC y *data augmentation*. Junto con ESC-50 y AudioSet, forma parte del canon de datasets que estructuran la enseñanza y la evaluación en ESC. Su legado metodológico —haber **normalizado publicar folds fijos** para garantizar comparaciones honestas— trasciende el audio.

## Limitaciones

- **Solo 10 clases.** Limitado por el costo de anotación; los autores lo plantean como punto de partida, no como cobertura exhaustiva.
- **Saliencia subjetiva.** Las etiquetas FG/BG son juicios de los anotadores, así que el efecto de la interferencia de fondo no puede cuantificarse rigurosamente.
- **Baseline deliberadamente simple.** El enfoque *bag-of-frames* con MFCC ignora la dinámica temporal y falla en sonidos continuos tipo ruido (confunde aires acondicionados con motores al ralentí, martillos con perforaciones).
- **Sesgo de origen.** Provenir de Freesound y del servicio 311 de Nueva York sesga tanto la selección de clases como las condiciones de grabación.
- **Tope de 1.000 fragmentos por clase.** Evita desbalances groseros pero no garantiza balance perfecto ni refleja la frecuencia real de cada sonido.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) presenta UrbanSound8K como dataset didáctico de ESC, y este paper es la fuente primaria de las cifras que la clase cita: **8.732 clips de $\leq 4$ s, 10 clases urbanas, 10 folds** (ver [datasets de audio](/fundamentos/datasets-de-audio)). Tres ideas conviene internalizar:

1. **Un benchmark es más que audio: es taxonomía + anotación + protocolo de evaluación.** La secuencia taxonomía → recolección desde Freesound → anotación con marcas de tiempo y saliencia → fragmentación en 4 s → folds fijos es una **plantilla reutilizable** para construir cualquier dataset de audio serio.
2. **Respetar los folds no es formalidad, es correctitud experimental.** Re-barajar con un split aleatorio produce exactitudes artificialmente altas e incomparables: usar los **10 folds tal como vienen** es exactamente lo que la clase enfatiza.
3. **El baseline MFCC fija el punto de partida.** Antes de las CNN espectrográficas, MFCC + SVM/random forest era el estado del arte práctico; conocerlo permite dimensionar cuánto aportó el deep learning sobre el mismo benchmark y protocolo.

La clasificación de sonidos ambientales que este dataset popularizó es directamente aplicable al **monitoreo acústico del hogar y a la asistencia a personas** en salud: un sistema entrenado para reconocer fuentes cotidianas puede detectar sonidos de alerta —una alarma, una caída, tos persistente, llanto— y disparar avisos en teleasistencia. La **anotación con saliencia** (primer plano vs. fondo) anticipa el desafío de distinguir el evento clínicamente relevante del ruido doméstico, y la disciplina de **folds sin fuga** es la que evita sobreestimar el desempeño de un clasificador de salud antes de desplegarlo.
