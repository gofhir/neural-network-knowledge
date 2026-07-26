---
title: "On the Measure of Intelligence — ARC (2019)"
weight: 379
math: true
---

{{< paper-card
    title="On the Measure of Intelligence"
    authors="François Chollet (Google)"
    year="2019"
    venue="arXiv:1911.01547"
    pdf="/papers/arc-chollet-2019.pdf" >}}
Chollet —creador de Keras— argumenta que la comunidad de IA ha estado midiendo lo incorrecto: durante décadas se equiparó inteligencia con **habilidad (skill)** en tareas específicas (ajedrez, Go, Atari, DotA2), pero la habilidad se puede **"comprar"** con recursos —basta con suficientes *priors* codificados a mano o suficientes datos para alcanzar cualquier nivel de desempeño sin exhibir generalización propia. Su tesis: la inteligencia es la **eficiencia en la adquisición de habilidades** sobre tareas nuevas que involucran novedad e incertidumbre, formalizada con Teoría de la Información Algorítmica. El aporte doble es esa redefinición y un benchmark concreto —el **Abstraction and Reasoning Corpus (ARC)**— construido sobre *priors* de **Core Knowledge** con tareas *few-shot* de transformación de grillas diseñadas para resistir la memorización. Es la prueba de referencia del [razonamiento](/fundamentos/razonamiento) genuino de la [Clase 34](/clases/clase-34).
{{< /paper-card >}}

---

## Contexto: qué mide realmente un benchmark

Chollet contrasta dos concepciones históricas de la inteligencia: como **colección de habilidades específicas** (psicología evolutiva, Minsky: la mente como ensamblaje de programas verticales) y como **capacidad general de aprendizaje** (*tabula rasa* de Locke y Turing: el cerebro del niño como cuaderno en blanco). Sostiene que **ambas son incorrectas**: la mente no es ni un conjunto de programas fijos ni una pizarra en blanco, sino un sistema con *priors* innatos que, lejos de limitar la generalización, **son su fuente**.

El problema práctico es la deriva de la evaluación: optimizar una sola métrica induce **atajos**, y la fijación en el desempeño de tareas específicas —"sin condición sobre cómo el sistema llega a ese desempeño"— produce sistemas que resuelven la tarea sin ser inteligentes. Chollet estructura el debate con un **espectro de generalización**: ausencia de generalización (tic-tac-toe por enumeración), **generalización local o "robustez"** (manejar puntos nuevos de una distribución conocida con muestra densa, lo que el ML hace desde los años 50), **generalización amplia o "flexibilidad"** (tareas no previstas por los creadores), y **generalización extrema** (tareas enteramente nuevas, solo lograda por formas biológicas). El defecto fatal de los benchmarks multi-tarea (GLUE, Arcade Learning Environment): **el conjunto de tareas es conocido de antemano** y el desarrollador puede entrenar para ellas —sigue midiendo habilidad, no capacidad.

## Contribución: definición formal + ARC

La tesis que Chollet formaliza:

> La inteligencia de un sistema es una medida de su eficiencia en la adquisición de habilidades sobre un alcance de tareas, con respecto a *priors*, experiencia y dificultad de generalización.

El andamiaje se monta sobre la **complejidad de Kolmogorov** $H(s)$ (longitud del programa más corto que produce $s$) y su versión relativa $H(s_1 \mid s_2)$. Separando el **sistema inteligente** (IS, que genera programas de habilidad) de los **programas de habilidad** (que ejecutan la conducta), define la **dificultad de generalización** como la fracción de la complejidad de la solución de evaluación no explicada por la mejor solución de entrenamiento:

$$GD^{\theta}_{T,C} = \frac{H\left(Sol^{\theta}_T \mid TrainSol^{opt}_{T,C}\right)}{H\left(Sol^{\theta}_T\right)}$$

Reuniendo *priors* ($P$, cuán cerca de una solución arranca el sistema), experiencia ($E$, información novedosa que reduce la incertidumbre) y dificultad, la inteligencia queda esquemáticamente como:

$$\mathbb{E}\left[\frac{\text{habilidad} \cdot \text{generalización}}{\text{priors} + \text{experiencia}}\right]$$

Cuando el denominador crece sin límite, el desempeño alto deja de ser evidencia de inteligencia. Chollet subraya que la alta habilidad **no** es alta inteligencia, y que la inteligencia **no** es *curve-fitting* (un sistema que solo produce el programa más simple consistente con los datos solo rinde en tareas de dificultad de generalización cero). Adopta explícitamente un marco **antropocéntrico**, distanciándose de la Inteligencia Universal de Legg-Hutter.

## Core Knowledge y el diseño de ARC

Si la inteligencia se mide controlando *priors*, hay que conocer los *priors innatos humanos*. Chollet recurre a la teoría de **Core Knowledge** (Spelke y Kinzler), cuatro sistemas innatos: **objetualidad y física elemental** (cohesión, persistencia, contacto), **agentividad y direccionalidad a metas**, **números naturales y aritmética elemental**, y **geometría y topología elementales**. Los distingue de los *priors de meta-aprendizaje*, que **no** se controlan porque **son** la inteligencia misma.

**ARC** instancia estas guías. Tiene **1.000 tareas únicas** (400 de entrenamiento, 600 de evaluación —400 públicas, 200 privadas— disjuntas del entrenamiento). Cada tarea presenta un puñado de **ejemplos de demostración** (3,3 en promedio) y generalmente **un** ejemplo de test; cada ejemplo es un par de **grillas** de hasta 30×30 con 10 símbolos-color. El examinado debe **construir desde cero** la grilla de salida (dimensiones, símbolos y posiciones), con éxito **binario** y **3 intentos** por test. Los *priors* de ARC son exactamente los de Core Knowledge, y todas las cantidades son menores a ~10. Las propiedades que resisten la memorización: mide **solo inteligencia fluida** (sin lenguaje ni conocimiento del mundo real), las tareas de evaluación son **únicas y desconocidas para los desarrolladores** (impide codificar la solución como programa —el fraude que arruinó los tests de CI a máquinas—), tiene **alta diversidad**, y las tareas son **generadas manualmente** (no programáticamente, evitando la ingeniería inversa de un programa maestro).

## Resultados y limitaciones

Cada tarea fue resuelta por al menos uno de tres humanos de CI alto sin comunicarse; un humano típico resuelve la mayoría **sin entrenamiento previo**. En contraste, hasta donde Chollet sabía en 2019, ARC **no parecía abordable por ninguna técnica de ML existente, incluido el Deep Learning**, precisamente por su foco en generalización amplia y aprendizaje *few-shot*.

Chollet reconoce que ARC es *work in progress*: la dificultad de generalización **no está cuantificada** (planea estimarla vía desempeño humano), la validez estadística no está establecida, el tamaño de 1.000 tareas puede ser vulnerable a atajos, y el formato binario 0/1 con 3 intentos es demasiado cerrado. Postula que un solucionador definiría un **DSL** que codifique los *priors* de Core Knowledge como funciones combinables —esto es, **síntesis de programas**—, advirtiendo que elegir simplemente el programa más simple que funciona en entrenamiento **no generaliza bien**.

## Por qué importa para la Clase 34

ARC ofrece a la [Clase 34](/clases/clase-34) un criterio riguroso para distinguir [razonamiento](/fundamentos/razonamiento) genuino de interpolación estadística sobre datos densamente muestreados:

- **Abstracción.** Resolver una tarea exige inferir la *regla abstracta* que gobierna los pares demostración (completar la simetría, extrapolar la línea que rebota, seleccionar el objeto más frecuente) a partir de poquísimos ejemplos. No hay superficie estadística que explotar: la señal está en la estructura, no en la textura.
- **Sistematicidad y generalización composicional.** El DSL de funciones base que deben **recombinarse** de formas novedosas para cada tarea es una prueba operacional de sistematicidad: entender "simetría" y "conteo" por separado debería bastar para componerlas de maneras no vistas.
- **Generalización sobre memorización.** La lección transversal —un sistema puede rendir alto sin razonar— queda formalizada: la habilidad es el artefacto cristalizado; la inteligencia es el proceso que la produce, medible solo controlando *priors* y experiencia.

El marco explica con precisión por qué los LLMs —con *priors* y experiencia casi ilimitados— tensionan pero no dominan ARC: cada tarea es única y con datos minúsculos por tarea, así que no hay muestreo denso que "comprar". Que **avancen** con andamiajes de síntesis de programas confirma el camino que Chollet señaló; que **no lo dominen** confirma que la interpolación estadística no equivale a abstracción composicional. Para un sistema clínico, la distinción entre **habilidad memorizada** y **eficiencia de adquisición** es directamente accionable: saturar un benchmark médico conocido no predice el comportamiento ante una presentación atípica, una comorbilidad rara o una población no representada —el análogo médico del *few-shot* de ARC.
