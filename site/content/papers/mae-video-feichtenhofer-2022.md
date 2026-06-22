---
title: "Masked Autoencoders As Spatiotemporal Learners (2022)"
weight: 327
math: true
---

{{< paper-card
    title="Masked Autoencoders As Spatiotemporal Learners"
    authors="Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, Kaiming He"
    year="2022"
    venue="NeurIPS 2022"
    pdf="/papers/mae-video-feichtenhofer-2022.pdf"
    arxiv="2205.09113" >}}
Extensión natural y deliberadamente minimalista de [MAE (He et al., 2022)](/papers/mae-he-2022) al **video**. La tesis es casi provocadora: no hace falta un método nuevo para aprender representaciones espaciotemporales [autosupervisadas](/fundamentos/aprendizaje-autosupervisado); basta tratar el video como un conjunto de **parches espaciotemporales** (tubelets de tamaño *t×16×16*), enmascarar al azar una fracción enorme de ellos y entrenar un autoencoder para reconstruir los píxeles faltantes. El hallazgo central: la tasa óptima de máscara en video sube a **90%** (frente al 75% de imágenes y al 15% de BERT en texto), porque el video añade una fuerte **redundancia temporal**. Esa tasa altísima no es curiosidad: es el motor de eficiencia (speedup real de 4.1× gracias al encoder solo-visible). En [Kinetics-400](/dominios/video) el pre-entrenamiento lleva la accuracy de 71.4% (desde cero) a 84.4%, **+13% absoluto**.
{{< /paper-card >}}

---

## Contexto: de BERT y MAE al video

El árbol genealógico es preciso. Las raíces están en los **denoising autoencoders** (Vincent et al., 2008): aprender representaciones reconstruyendo una señal limpia a partir de una entrada corrompida, donde el enmascaramiento es un tipo de corrupción. **BERT** (2019) es masked autoencoding sobre tokens de lenguaje, el éxito que legitimó la idea a gran escala. En visión la línea progresó por etapas: **iGPT** trató píxeles como tokens; **ViT** dio el salto a usar *parches* como tokens; y **MAE** (He et al., 2022) volvió a los fundamentos del autoencoding poniendo el foco en el **decoder** y en el **encoder asimétrico solo-visible**.

Este trabajo hereda esa receta casi sin cambios y la transporta al [video](/dominios/video). El campo del autosupervisado en video venía dominado por familias muy diferenciadas: coherencia temporal o *slowness*, predicción de futuro, movimiento de objetos, ordenamiento temporal y contraste espaciotemporal. El método aquí presentado **también** explota la coherencia temporal, pero de forma **implícita**: como es prácticamente agnóstico al espacio-tiempo, la única vía por la que la aprovecha es subiendo la tasa de máscara al 90%, lo que presupone que el video es más redundante que la imagen. El paper reconoce ser concurrente e independiente de **VideoMAE** (Tong et al., 2022).

## La idea: MAE sobre parches espaciotemporales

"In a nutshell, simply MAE applied to the set of spacetime patches." La contribución no es un mecanismo nuevo sino una **demostración**: que MAE aplicado a la rejilla de tubelets produce representaciones de video muy fuertes con sesgos inductivos mínimos. Los pilares:

1. **Enmascaramiento espaciotemporal al 90%.** Se muestrean parches al azar (sin reemplazo) de la rejilla *T×H×W*. La tasa óptima sube de 75% (imagen) a **90%** (video), e incluso 95% rinde sorprendentemente bien. La justificación es un experimento mental: si un video tuviera *T* fotogramas idénticos, muestrear *1/T* de los parches ya revelaría casi todo el fotograma estático; como en videos naturales el movimiento lento es más probable que el rápido, la tasa puede ser altísima.

2. **Muestreo agnóstico al espacio-tiempo.** El muestreo aleatorio *no* respeta la estructura espaciotemporal, análogo a BERT en 1D y MAE en 2D. Esto **supera** a las alternativas estructuradas —space-only ("tube"), time-only ("frame") y block-wise ("cube")— porque estas, con tasas muy altas, dejan tareas demasiado fáciles o imposibles (p. ej., conservar un solo fotograma exige predecir pasado y futuro desde una imagen).

3. **Encoder asimétrico solo-visible.** Con 90% de máscara el encoder ve <1/10 de los tokens. Como la auto-atención es cuadrática en el número de tokens, esto reduce el cómputo del encoder a <1/10; sumando un decoder pequeño da una reducción teórica de **7.7× en FLOPs** y un **speedup real de 4.1×**.

4. **Arquitectura agnóstica.** Encoder y decoder son **ViT vanilla** sin factorización ni jerarquía, en contraste con los líderes especializados (ViViT, MViT, Video Swin). El único componente *spacetime-aware* es el embedding de parches; el método predice píxeles, sin tokenizer específico.

## Método

**Patch embedding (el único componente con conocimiento de dominio).** Siguiendo ViT, el clip se divide en una rejilla regular de parches no solapados de **2×16×16**. Para una entrada de **16×224×224** eso produce **8×14×14 = 1568 tokens**. Los parches se aplanan, se proyectan y se les suman **embeddings posicionales separables** (uno para el espacio, otro para el tiempo), cuya suma da el embedding espaciotemporal; esta separación evita que el embedding posicional crezca demasiado en 3D.

**Enmascaramiento.** Con la tasa del 90%, de los 1568 tokens solo **156 quedan visibles**. El muestreo agnóstico gana en las ablaciones (Kinetics-400, ViT-L, 800 épocas):

| Estrategia | Tasa | Acc. K400 |
|---|---|---|
| **Agnóstica (random)** | **90%** | **84.4%** |
| Space-only ("tube") | 90% | 83.5% |
| Time-only ("frame") | 75% | 79.1% |
| Block-wise ("cube") | 75% | 83.2% |

**Autoencoding.** El **encoder** (ViT vanilla) opera solo sobre los parches visibles. El **decoder** (otro ViT vanilla, deliberadamente más pequeño: 512-d, 4 bloques, frente al encoder ViT-L de 1024-d, 24 bloques) procesa la unión del conjunto codificado más **tokens de máscara**. La **predicción** es en el espacio de píxeles —basta predecir un único corte temporal (16×16) por tubelet—, sobre píxeles **normalizados por parche** (+0.6%). La **pérdida** es MSE sobre los parches desconocidos, como en BERT/MAE. No hay jerarquía ni factorización: el método confía en la auto-atención **global** para aprender desde los datos.

**Implementación.** Entrada por defecto 16×224×224 (16 fotogramas a stride temporal 4). Como el pre-entrenamiento es tan rápido que la carga de datos pasa a ser el cuello de botella, se adopta **repeated sampling** (4 muestras por video cargado), que sube la velocidad de pared hasta 3×. La evaluación se hace por **fine-tuning end-to-end**, no linear probing.

## Experimentos

**Kinetics-400.** Con ViT-L vanilla, el pre-entrenamiento MAE de 800 épocas lleva la accuracy de **71.4% (desde cero) a 84.4%**, un salto **absoluto de +13.0%**. Esa brecha es mucho mayor que en imágenes (~3% en MAE original), lo que sugiere que el pre-entrenamiento MAE es **especialmente útil para video**. La eficiencia (ViT-L, 800 épocas):

| Encoder | FLOPs | Cómputo | Carga+cómputo |
|---|---|---|---|
| Denso (con [M]) | 627.5 G | 141.1 h | 147.5 h |
| **Sparse (sin [M])** | **81.0 G** | **24.5 h** | **35.8 h** |
| Ganancia | **7.7×** | 5.8× | **4.1×** |

**Tasa de máscara.** El 90% es la mejor; el 95% rinde sorprendentemente bien. Las tasas bajas (75%, 50%) rinden **peor pese a ver más tokens y costar más** —el 75% óptimo en imágenes no lo es en video—, lo que respalda la hipótesis de mayor redundancia. A diferencia de imágenes, un decoder demasiado estrecho o poco profundo degrada notablemente: el video, más complejo, exige más capacidad de decodificación.

**Datos reales no curados.** Pre-entrenar con K400 (video) bate al supervisado por márgenes enormes: **+9.5% en AVA** (detección de acción) y **+16.4% en SSv2**. Más sorprendente aún: pre-entrenar con **1 millón de videos de Instagram no curados** rinde casi igual que datos curados —comportamiento que **NO se observa en métodos contrastivos**, donde la curación importa mucho. MAE es robusto a la distribución de datos. A nivel de sistema es competitivo con los líderes (ViT-H alcanza 85.1% en K400) siendo **la única entrada líder basada en ViT vanilla**.

## Limitaciones reconocidas

La conclusión es explícitamente modesta. (i) **Escala de datos:** lo explorado es órdenes de magnitud menor que las contrapartes de lenguaje (GPT-3); el video de alta dimensión sigue siendo un reto mayor para escalar. (ii) El paper se reporta solo como **señal inicial** para investigación futura. (iii) El método **no aporta un mecanismo temporal explícito** —la coherencia temporal se explota implícitamente vía la alta tasa de máscara— y, al ser agnóstico a la estructura, cede algo de precisión frente a arquitecturas jerárquicas especializadas a igual resolución, compensándolo con simplicidad y eficiencia.

## Por qué importa

El valor histórico es triple. Primero, **completa la tríada del masked autoencoding** —lenguaje (BERT), imagen (MAE), video (este paper)— sosteniendo empíricamente que un mismo principio generativo, "reconstruir lo enmascarado", sirve para los tres medios con conocimiento de dominio mínimo. Segundo, ofrece una **lectura cuantitativa de la redundancia** vía la tasa óptima de máscara (15% → 75% → 90%), intuición que se volvió canónica para razonar sobre [SSL generativo](/fundamentos/aprendizaje-autosupervisado) en cualquier modalidad. Tercero, su **eficiencia** (encoder solo-visible + 90% de máscara → 4.1× de speedup) hizo *práctico* el pre-entrenamiento autosupervisado de [video](/dominios/video) a gran escala, un dominio notoriamente caro.

Junto con VideoMAE, definió el paradigma de los *video masked autoencoders* que dominó el SSL de video posterior, desplazando en buena medida a los métodos contrastivos —en parte gracias a su robustez a datos no curados. Para dominios donde la augmentación es difícil o inválida (imagen médica, sensado remoto, datos geométricos y sus extensiones temporales), su naturaleza generativa y poco dependiente de augmentación lo hace generalizable.

## Conexión con la Clase 28

El temario de la [Clase 28](/clases/clase-28) (Aprendizaje Autosupervisado) coloca el slide "MAE en videos" **inmediatamente después** de la presentación de MAE para imágenes, y esa secuencia es exactamente la narrativa del paper. La lección pedagógica central: pasar de imagen a video **no requirió un método nuevo** —solo cambiar la rejilla de parches 2D por tubelets 3D y subir la tasa de máscara—, lo que ilustra de forma nítida la idea de un marco unificado de SSL generativo con sesgos inductivos mínimos. Tres puentes concretos:

- **La tasa de máscara como medidor de redundancia.** El salto 75% → 90% es el ejemplo más limpio del curso para enseñar *por qué* funciona el masked autoencoding: cuanto más redundante el medio, más se puede enmascarar y más útil se vuelve la tarea de pretexto. Conecta con el fundamento de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).
- **Continuidad con MAE de imágenes.** El paper reutiliza casi literalmente la receta de [He et al. (MAE), 2022](/papers/mae-he-2022): estudiar ambos en orden muestra qué se mantiene (la arquitectura asimétrica) y qué cambia (la tasa, el embedding separable espacio/tiempo, el decoder algo mayor).
- **SSL generativo vs. contrastivo en video.** La robustez de MAE a datos de Instagram no curados —donde el contrastivo se degrada— es un argumento empírico fuerte a favor de los métodos generativos para datos reales a escala.
