---
title: "Show Your Work: Scratchpads (2021)"
weight: 381
math: true
---

{{< paper-card
    title="Show Your Work: Scratchpads for Intermediate Computation with Language Models"
    authors="Maxwell Nye et al. (MIT, Google)"
    year="2021"
    venue="arXiv:2112.00114"
    pdf="/papers/scratchpad-nye-2021.pdf" >}}
Los grandes modelos de lenguaje rinden asombrosamente bien en tareas resolubles "de una sola pasada" (generar texto, sintetizar programas cortos), pero **fracasan en cómputo multi-paso no acotado**: sumar enteros largos o predecir el resultado de ejecutar un programa. La propuesta es tan simple como poderosa: en vez de exigir la respuesta final de golpe, se le permite al modelo **producir una secuencia arbitraria de tokens intermedios —un "scratchpad" o bloc de notas— antes de la respuesta**. En ese bloc escribe los pasos del algoritmo (los acarreos de una suma, los términos de un polinomio, la traza línea por línea de un programa). Entrenando Transformers con supervisión de máxima verosimilitud para que emitan esos pasos, el desempeño en aritmética, evaluación de polinomios y ejecución de código **mejora drásticamente**. Es el **precursor directo de [Chain-of-Thought](/fundamentos/chain-of-thought)** (Wei et al., 2022) y una pieza de bisagra de la [Clase 34](/clases/clase-34).
{{< /paper-card >}}

---

## Contexto: por qué los LLMs fallan en cómputo multi-paso

Los Transformers grandes escriben código que resuelve problemas simples de programación, pero tropiezan de forma sistemática con cálculos algorítmicos multi-paso. El paper enumera síntomas concretos: **GPT-3 no puede sumar en few-shot** números de más de tres dígitos; los modelos **no logran predecir el resultado de ejecutar código Python**, ni siquiera el que ellos mismos escriben; las redes recurrentes y de grafos **no generalizan sistemáticamente** al predecir la salida de programas con bucles. De aquí una observación filosóficamente interesante: los modelos pueden **escribir** código, pero no parecen **representar su semántica**, porque no pueden predecir su ejecución. Saben la forma, no el significado.

La hipótesis de los autores es arquitectónica, no de capacidad bruta. El problema está en **cómo se aplica** el Transformer: se le pide realizar el cómputo **en una sola pasada hacia adelante** (*a single forward pass*). Con un número fijo de capas y de cómputo, el modelo **no puede adaptar el esfuerzo según la dificultad** antes de producir la salida: una suma de 3 dígitos y otra de 30 reciben el mismo presupuesto. Trabajos previos atacaron esto **modificando la arquitectura** (Adaptive Computation Time, PonderNet, Universal Transformers). La apuesta de este paper es contraria y más pragmática: **no tocar el modelo, sino rediseñar la tarea**.

## Contribución y método: el scratchpad

La propuesta se resume en una frase: **permitir que el modelo produzca una secuencia arbitraria de tokens intermedios antes de la respuesta final**, codificando como texto los pasos del algoritmo y usando entrenamiento supervisado estándar. Esta modificación aporta cuatro ventajas articuladas con precisión: **cómputo adaptativo** (un problema difícil genera un scratchpad más largo y recibe más pasos de cómputo); **memoria externalizada** (el estado intermedio vive en el buffer y se reconsulta atendiendo al propio contexto, en vez de comprimirse en las activaciones); **cuantización de errores** (los estados quedan fijados a embeddings de tokens, frenando la propagación de errores pequeños típica de las Neural Turing Machines); e **interpretabilidad** (examinar el scratchpad ayuda a identificar y corregir errores de formato).

El formato es concreto. Para sumar 29 y 57, el objetivo delimitado por `<scratch>` … `</scratch>` registra el algoritmo escolar dígito a dígito, con el acarreo tras `C:`; se **introducen espacios entre dígitos** para que cada uno sea un token separado y el tokenizador no arruine la aritmética. Para ejecución de programas, el objeto clave es la **traza**: la secuencia alternada de líneas de código ejecutadas y el estado de las variables locales tras cada línea, con **todos los bucles desplegados (*unrolled*)** —justo lo que da el cómputo adaptativo. Todos los experimentos usan **Transformers densos decoder-only pre-entrenados**, de **2 millones a 137 mil millones de parámetros**.

## Resultados

- **Adición de enteros largos.** Más allá de un tamaño crítico de modelo, los modelos con scratchpad resuelven la suma; los entrenados sin él fracasan incluso a la mayor escala. En las pruebas **fuera de distribución** (9 y 10 dígitos), la línea base directa fracasa por completo, mientras el scratchpad **generaliza a longitudes no vistas** —lo que uno espera de un modelo que aprendió el algoritmo en vez de memorizar respuestas. Un ablation revelador: si al modelo con scratchpad se le hace fine-tuning para responder directo, no hay mejora sobre la base; es el uso del scratchpad **en inferencia** lo que importa.
- **Evaluación de polinomios** (grado $\leq 3$). El scratchpad supera a la predicción directa en ambos regímenes: en **few-shot** (137B) más que duplica el desempeño, de **8.8 % a 20.1 %**; en **fine-tuning** (8B) lo eleva de **31.8 % a 50.7 %**.
- **Ejecución de programas Python**, el resultado más ambicioso porque **subsume** a los anteriores. En programas sintéticos, few-shot pasa de 11 % a **26.5 %** y fine-tuned de 20 % a **41.5 %**. En programas reales (MBPP), el aumento de datos con programas muestreados (MBPP-aug) **daña** la ejecución directa pero **mejora enormemente** al scratchpad, porque los candidatos, aun incorrectos, siguen siendo **trazas de ejecución válidas**. El mejor modelo combinado (MBPP-aug + CodeNet + single-line) traza perfectamente **24.6 %** de las tareas, y la decodificación greedy produce la traza exactamente correcta para **casi el 42 %** de todas las trazas.

## Limitaciones

- **Ventana de contexto.** Todo se limita a scratchpads que caben en la ventana de generación (512 tokens); en MBPP esto reduce el conjunto evaluable de 500 a 212 tareas. Realizar plenamente el potencial de la técnica exige ventanas más largas.
- **Supervisión con trazas.** En la inducción de algoritmos hay que **diseñar a mano** los estados intermedios de cada tarea, algo que los propios autores reconocen sub-óptimo (la ejecución de programas lo alivia porque las trazas se generan con un intérprete).
- **Aprender a usar el scratchpad sin supervisión** —por ejemplo con RL, recompensando respuestas correctas con menos tokens de scratchpad— queda como siguiente paso. Este es, de hecho, el germen de la investigación posterior en razonamiento auto-inducido.

## Por qué importa para la Clase 34

En la [Clase 34](/clases/clase-34) (Razonamiento) el scratchpad se presenta como el andamiaje que fuerza al modelo a exteriorizar el cómputo en vez de comprimirlo en una sola pasada opaca. El título —*Show Your Work*, "muestra tu trabajo"— es la instrucción que todo profesor de matemáticas da a un alumno: no me des solo el número, muéstrame cómo llegaste. A los Transformers les pasa lo mismo: **el acto de escribir los pasos es lo que hace posible resolver el problema**.

La relación con [Chain-of-Thought](/papers/chain-of-thought-wei-2022) (Wei et al., 2022) es la del precursor con su sucesor. Ambos comparten la tesis central —darle al modelo espacio para generar pasos intermedios transforma problemas irresolubles de una sola pasada en resolubles— pero difieren en **cómo se obtiene el comportamiento**: Nye et al. lo logran por **fine-tuning con trazas explícitas**, tocando los pesos, sobre cómputo algorítmico estructurado; Chain-of-Thought lo consigue un año después **solo por prompting**, sin tocar los pesos, sobre razonamiento en lenguaje natural, aprovechando que los modelos suficientemente grandes ya tienen la habilidad latente. Chain-of-Thought es, en un sentido preciso, "el scratchpad sin fine-tuning". Ambos son ancestros del [razonamiento](/fundamentos/chain-of-thought) moderno —*zero-shot CoT*, *self-consistency*, modelos de razonamiento con cómputo de inferencia extendido—: descendientes de la intuición que este paper formalizó primero, **el cómputo intermedio explícito es lo que separa a un LLM que adivina de uno que razona**.
