---
title: "SWAG: Adversarial Dataset for Commonsense Inference (2018)"
weight: 301
math: true
---

{{< paper-card
    title="SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference"
    authors="Rowan Zellers, Yonatan Bisk, Roy Schwartz, Yejin Choi (University of Washington / Allen Institute for AI)"
    year="2018"
    venue="EMNLP 2018 / arXiv:1808.05326"
    arxiv="1808.05326"
    pdf="/papers/swag-zellers-2018.pdf" >}}
113.000 preguntas de opción múltiple sobre qué pasa después de una escena de video. Su aporte duradero no es el dataset sino **Adversarial Filtering**: un procedimiento para generar distractores que resisten a los modelos del momento, filtrando iterativamente los que resultan demasiado fáciles. La ironía es que el dataset construido para durar **sobrevivió tres meses**: [BERT](/papers/bert-devlin-2018) lo superó en noviembre de 2018 y cruzó el desempeño humano experto. Es la tarea que el [Laboratorio 20](/laboratorios/lab-20) toca al cargar `XLNetForMultipleChoice`.
{{< /paper-card >}}

---

## La tarea

Dada una premisa tomada de subtítulos de video —*"Ella abre el capó del auto"*— elegir cuál de cuatro continuaciones es la real. Los distractores no son aleatorios: son generados por un modelo de lenguaje y **filtrados para que sean difíciles**.

Lo que la tarea intenta medir es *sentido común situado*: no conocimiento enciclopédico, sino saber qué suele ocurrir a continuación en el mundo físico.

## Adversarial Filtering

Esta es la contribución metodológica, y la que se reutilizó después en HellaSwag, WinoGrande y buena parte de los benchmarks de la generación siguiente.

Un dataset es **adversarial respecto de una familia de modelos** $f$ si, para cualquier partición, el error esperado se mantiene alto:

$$I(D, f) = \frac{1}{N} \sum_{i=1}^N L\left(f_{\theta_i^\star}, \{(x_i, y_i)\}\right), \qquad \theta_i^\star = \arg\min_\theta L\big(f_\theta,\, D \setminus \{(x_i, y_i)\}\big)$$

Optimizar eso directamente es intratable, así que el algoritmo procede por iteración:

```
mientras no converja:
    1. Partir D en train/test ficticio (80/20)
    2. Entrenar un ensemble sobre train
    3. Para cada ejemplo en test:
       a. Identificar los distractores "faciles" — los que el ensemble
          separa claramente del correcto
       b. Reemplazarlos por candidatos del pool que el modelo clasifica MAL
```

La intuición: un distractor que el modelo descarta con facilidad está siendo descartado **por estilo, no por contenido** — longitud, frecuencia léxica, fluidez. Reemplazarlo por uno que confunde al modelo elimina ese atajo.

{{< concept-alert type="clave" >}}
Adversarial Filtering no busca hacer la tarea más difícil en abstracto: busca **eliminar los atajos estadísticos** que permiten resolverla sin entenderla. Es el mismo problema que aparece en el propio Lab 20 con el dataset de *fake news*, donde las noticias reales vienen de Reuters y las falsas de sitios diversos: un clasificador puede alcanzar accuracy alta aprendiendo el **estilo de redacción de Reuters**, sin evaluar factualidad alguna.

La diferencia es que SWAG intenta cerrar esa puerta por construcción, y el dataset del laboratorio la deja abierta de par en par.
{{< /concept-alert >}}

## Los tres meses de vida útil

| Sistema | Test accuracy |
|---|---:|
| ESIM + GloVe (el baseline del paper) | 52,7 |
| ESIM + [ELMo](/papers/elmo-peters-2018) | 59,2 |
| [GPT](/papers/gpt-1-radford-2018) fine-tuneado | 78,0 |
| **BERT-base** | **81,6** |
| **BERT-large** | **86,3** |
| Humano experto | 85,0 |
| Humano (mayoría de 5) | 88,0 |

**BERT-large superó al humano experto por 1,3 puntos**, tres meses después de la publicación. El dataset quedó esencialmente saturado.

La explicación está en algo que los propios autores anticipaban (§5.4): el ensemble usado para el filtrado eran modelos **estilísticos y léxicos** —perplejidad, bolsa de palabras, CNN, BiLSTM sobre etiquetas POS—. Esos capturan artefactos de superficie, pero no representaciones contextuales pre-entrenadas sobre miles de millones de tokens.

BERT no es "el mismo tipo de modelo, más grande". Trae conocimiento de mundo del pre-entrenamiento, atención cruzada bidireccional entre premisa y candidato a lo largo de 24 capas, y un objetivo —completar huecos según contexto— que es casi exactamente la habilidad que SWAG mide.

La lección que quedó, y que el paper de HellaSwag (2019) hizo explícita al usar BERT *como filtro*: **Adversarial Filtering solo garantiza dificultad frente a la familia de modelos con la que se filtró**. Es una propiedad relativa, no absoluta, y por eso los benchmarks construidos así envejecen tan rápido como el estado del arte.

## Limitaciones reconocibles

- **Dificultad relativa al filtro**, como se acaba de ver. HellaSwag lo corrige subiendo la potencia del ensemble, con lo que hereda el mismo techo desplazado.
- **Sesgo de dominio.** Las premisas vienen de subtítulos de video (ActivityNet y LSMDC): escenas de actividades cotidianas filmadas. El "sentido común" que mide es el de ese dominio.
- **Los distractores los genera un modelo de lenguaje**, así que el dataset hereda las regularidades de ese generador — un tipo de contaminación difícil de auditar.
- **Erratas en la tabla 1** del paper, detectadas al reconstruir los números del análisis original.

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) carga `XLNetForMultipleChoice`, cuya cabeza está diseñada para esta forma de tarea: se codifica cada par (premisa, candidato) por separado, se extrae un escalar por candidato y se aplica softmax sobre las opciones.

Es la tercera de las tres variantes de XLNet que el notebook instancia —backbone, QA y opción múltiple—, y sirve para ver que las **cabezas de tarea son intercambiables sobre el mismo encoder**: lo que cambia entre ellas son unos pocos parámetros al final, no el modelo.

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [XLNet](/papers/xlnet-yang-2019) · [ELMo](/papers/elmo-peters-2018) · [SQuAD](/papers/squad-rajpurkar-2016) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
