---
title: "Gato: A Generalist Agent (2022)"
weight: 375
math: true
---

{{< paper-card
    title="A Generalist Agent"
    authors="Scott Reed, Konrad Żołna, Emilio Parisotto, et al. (DeepMind)"
    year="2022"
    venue="TMLR 2022 / arXiv:2205.06175"
    pdf="/papers/gato-reed-2022.pdf" >}}
Gato es un **único agente generalista**: una sola red neuronal con **el mismo conjunto de pesos** que —según cómo esté configurado su contexto— juega Atari, subtitula imágenes, conversa, apila bloques con un brazo robótico real y navega en entornos 3D. La tesis es tomar el enfoque que funcionó en modelado de lenguaje ([transformers](/papers/attention-is-all-you-need-vaswani-2017) autorregresivos sobre secuencias de tokens) y **extenderlo más allá del texto**, hacia una política multimodal, multitarea y multi-encarnación. Fue entrenado sobre **604 tareas** con apenas **1.2 mil millones de parámetros** (1.2B), tamaño elegido para permitir control en tiempo real de un robot físico. Para la [Clase 33](/clases/clase-33) es la evidencia (slide 33) de que **"el [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion) funciona bastante bien... aunque tampoco excelente"**: Gato es **behavioral cloning a escala masiva**, y no supera a los maestros que generaron sus datos —el techo estructural de la imitación pura.
{{< /paper-card >}}

---

## Contexto: por qué la Clase 33 cita a Gato

La [Clase 33](/clases/clase-33) contrasta dos familias de métodos para obtener políticas de control: el **aprendizaje reforzado**, que optimiza una recompensa escalar por ensayo y error y puede descubrir comportamientos nunca demostrados; y el [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion), que no optimiza recompensa sino que aprende de demostraciones de un experto tratando el problema como aprendizaje supervisado —predecir la acción experta dado el estado—. Su forma más simple es la **clonación de comportamiento (behavioral cloning, BC)**.

Gato encaja de lleno en la segunda familia. El paper lo dice sin ambigüedad: *"Gato is a data-driven approach, as it is derived from imitation learning"*. Su régimen es **puramente supervisado, offline**. La lección que la clase extrae es doble y matizada:

- **"Funciona bastante bien":** con una sola red de 1.2B parámetros, Gato rinde por sobre el 50 % del experto en **más de 450 de las 604 tareas**, incluyendo dominios tan dispares como Atari, subtitulado y manipulación robótica real.
- **"Tampoco excelente" / "no supera a su maestro":** los agentes de RL en línea que generaron los datos siguen **superando a Gato**; un especialista del mismo dominio lo supera; y Gato nunca excede sistemáticamente al experto que lo enseñó. Ese es el **techo del maestro** intrínseco al BC.

## Método y contribución

La contribución no es un algoritmo nuevo, sino una **demostración de existencia**: probar que un solo transformer, con los mismos pesos, entrenado como modelo de secuencias, puede ser generalmente competente en cientos de tareas heterogéneas sin sesgos inductivos por dominio. Tres ideas la sostienen:

**Tokenización unificada.** Todo se serializa en una única secuencia plana de enteros: **texto** con SentencePiece (32.000 subpalabras); **imágenes** en parches $16\times 16$ en orden raster (estilo ViT); valores **discretos** (botones) aplanados row-major; valores **continuos** (propiocepción, torques) codificados con mu-law y discretizados en 1024 bins. Cada paso temporal se representa como *observación → separador → acción*, y cada episodio como sus pasos en orden cronológico.

**Un solo modelo, los mismos pesos.** No es "una arquitectura con pesos distintos por tarea" (lo habitual en RL multitarea), sino **una sola red con los mismos pesos para todo**. Es un transformer decoder-only (24 capas, dimensión 2048), y es Gato quien decide, según su contexto, si emitir texto, torques o pulsaciones de botón.

**Entrenamiento = imitación masiva.** El objetivo es el mismo de un modelo de lenguaje —predecir el siguiente token:

$$\log p_\theta(s_1, \dots, s_L) = \sum_{l=1}^{L} \log p_\theta(s_l \mid s_1, \dots, s_{l-1})$$

Lo que lo convierte en imitación es la **función de enmascaramiento** $m(b, l)$: la pérdida solo penaliza tokens que son texto o la **acción registrada del agente experto**, no las observaciones.

$$\mathcal{L}(\theta, B) = - \sum_{b=1}^{|B|} \sum_{l=1}^{L} m(b, l)\, \log p_\theta\!\left(s_l^{(b)} \mid s_1^{(b)}, \dots, s_{l-1}^{(b)}\right)$$

Es behavioral cloning en su forma más pura, envuelto en la maquinaria de un modelo de secuencias: no hay recompensa en la función objetivo, no hay bootstrapping de Bellman, no hay exploración. Los datos de control provienen de **agentes especialistas de RL ya entrenados**, filtrados a episodios con al menos 80 % del retorno experto: **Gato imita a políticas de RL** que hicieron el trabajo duro de descubrir buenos comportamientos.

## Resultados

- **Amplitud:** por sobre el 50 % del puntaje experto en **más de 450 de las 604 tareas** con un único conjunto de pesos. En Atari alcanza el puntaje humano promedio en 23 juegos, pero los agentes de RL que generaron los datos **lo siguen superando**.
- **Robótica real (RGB Stacking, brazo Sawyer):** competitivo con el baseline publicado; iguala la performance de un especialista de BC entrenado solo para apilar (75.6 % vs. 74.6 % en Skill Mastery).
- **Escala:** con tres tamaños (79M, 364M, 1.18B), mejora consistente al aumentar la capacidad —la misma dinámica de los LLM.
- **Transferencia:** en RGB Stacking recupera la performance del experto con solo **10 episodios** de fine-tuning; en una tarea perceptual nueva, fine-tuneado con 500 demostraciones logra 60 % de éxito frente a 0.5 % de un BC desde cero.
- **Los especialistas ganan:** un modelo dedicado solo a Atari logra performance sobrehumana en 44 juegos, frente a los 23 del generalista.

## Limitaciones

- **El techo del maestro.** Gato imita trayectorias filtradas al 80 % del retorno experto y no usa la recompensa para mejorar más allá de lo demostrado: su rendimiento **está acotado por la calidad de los datos**. Los autores señalan que esto podría superarse con RL offline en lugar de supervisión pura.
- **No supera a los especialistas.** La generalidad tiene un costo de competencia por tarea.
- **Contexto corto (1024 tokens).** Para entornos con imágenes equivale a muy pocos pasos; el in-context learning en entornos nuevos **no mostró mejora significativa**, por lo que la adaptación se hace por fine-tuning, no por prompting.
- **Diálogo y captioning rudimentarios**, atribuidos a la escala moderada.

## Por qué importa para la Clase 33

Gato es el ejemplo canónico de **imitación a escala** en el marco de la [Clase 33](/clases/clase-33):

- **Es imitación, no RL.** Aunque sus datos provienen de agentes de RL y contienen recompensas, Gato **no optimiza la recompensa**; la usa apenas como filtro de calidad. Es behavioral cloning de manual, con la diferencia de escala y multimodalidad.
- **Ilustra la fortaleza del BC:** simple, estable y sample-efficient cuando hay buenas demostraciones, sin interacción con el entorno ni la tríada mortal del RL profundo.
- **Ilustra su debilidad estructural:** hereda el **techo del maestro** y sufre **cambio de distribución** (los errores se componen al llevar al agente a estados nunca demostrados). La clase propone remedios que Gato no aplica —**DAgger** (consultar al experto en los estados que visita el aprendiz) o el salto al **aprendizaje reforzado inverso / GAIL**, que *infieren la función de recompensa* del experto en lugar de copiar acciones, permitiendo en principio superar al maestro.

Gato encarna la frase de la clase con precisión: **la imitación funciona bastante bien, pero no excelente**, porque copiar tiene por techo la calidad de lo copiado. Es, además, la extensión directa del paradigma de los [transformers](/papers/attention-is-all-you-need-vaswani-2017) al control encarnado y un precursor de los modelos Vision-Language-Action posteriores.
