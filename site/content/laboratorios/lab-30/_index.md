---
title: "Lab 30 - Modelos con memoria externa: Key-Value Memory Networks"
weight: 300
sidebar:
  open: true
---

**Profesor:** Andrés Espinosa
**Fecha:** Junio 2026
**Notebook origen:** `clase_30/material/Laboratorio/Práctico_Memoria_externa_Final.ipynb`
**Notebook ejecutado:** [lab30.ipynb](/notebooks/lab30.ipynb) · [HTML](/notebooks-html/lab30.html)

## Encuadre

La contraparte práctica de la [clase 30](/clases/clase-30): implementar de cero el modelo **Key-Value Memory Network** ([Miller et al. 2016](/papers/key-value-memnn-miller-2016)) y aplicarlo a **Question Answering** sobre **WikiMovies** (preguntas y respuestas sobre películas, parte de los benchmarks [bAbI](/papers/babi-weston-2015)).

KV-MemNN es el eslabón **Memory Networks → End-to-End MemNN → Key-Value MemNN → Entity Networks** que separa cada hecho de la memoria en dos vistas:

- **Key** → lo que se usa para *direccionar* (matchea con la pregunta vía atención).
- **Value** → lo que se *lee/devuelve* cuando esa entrada gana atención.

Esto desacopla "cómo encuentro la información" de "qué información devuelvo". El lab construye la base de conocimiento **directamente desde texto de Wikipedia** (de ahí el "Directly Reading Documents" del título del paper), no desde una KB estructurada.

| Pieza | Implementación en el lab |
|---|---|
| Base de conocimiento | ventanas de ±3 tokens alrededor de cada entidad → entradas `(key, value)` duales (`__movie__` / `__window_center__`) |
| Candidate generation (blocking) | índice invertido token→entradas, con filtro de stopwords (`freqs < 1000`) |
| Codificación | Bag-of-Words: embedding de cada entrada = promedio de embeddings de sus tokens |
| Controlador | 2 hops de atención `key addressing → value reading → update R(q+o)` |
| Salida | scoring sobre los **candidatos recuperados** (no sobre el vocabulario completo) |
| Evaluación | top-1 accuracy sobre los ejemplos respondibles (~69% con el preprocesamiento óptimo) |

## Resultados consolidados

| Métrica (test, muestra aleatoria de 2000) | Valor |
|---|---|
| Top-1 accuracy | 0.683 |
| Top-3 accuracy | 0.741 |
| Top-5 accuracy | 0.799 |

### Las lecciones del lab

1. **El "31% de error" es engañoso.** Una parte sustancial no son errores de capacidad, sino **ground-truth single-answer** sobre preguntas multi-respuesta: en `describe X`, el modelo recupera todas las facetas válidas (director, género, año, actores) pero la métrica top-1 solo acepta la que el preprocesamiento fijó arbitrariamente (`first_answer`).
2. **El salto top-1→top-3 de +6 a +16 puntos** (según el slice) cuantifica directamente ese efecto multi-respuesta.
3. **El modelo está sobreconfiado** (softmax ~1.0 en un candidato): el producto punto `q·candidato` sin temperatura satura el softmax → problema de calibración, no de conocimiento.
4. **Dos modos de fallo distintos:** multi-respuesta (`describe`) y desambiguación de entidad homónima (`who directed heat?` → confunde el *Heat* de Michael Mann con el de Paul Morrissey).
5. **La memoria es no-paramétrica:** el modelo responde una pregunta inventada (`who directed doomsday?` → `neil marshall`) que nunca estuvo en ningún split, sin reentrenar — siempre que los tokens ya tengan embeddings entrenados.

## Bloques del lab

{{< cards >}}
  {{< card link="01-arquitectura-kvmemnn" title="Arquitectura del KVMemoryReader" subtitle="Embeddings compartidos, Bag-of-Words, key addressing + value reading, los 2 hops, scoring sobre candidatos" icon="code" >}}
  {{< card link="02-construccion-kb" title="Construcción de la KB y blocking" subtitle="Ventanas key/value duales, prefijo 1:, índice invertido, candidate generation, el truco del padding" icon="cube-transparent" >}}
  {{< card link="03-experimentos-y-analisis" title="Experimentos propios y análisis" subtitle="5 experimentos: visualización de atención, búsqueda de errores, top-5 candidatos, top-k accuracy, dataset ordenado" icon="beaker" >}}
  {{< card link="04-actividades" title="Actividades 1-4 resueltas" subtitle="Mejores embeddings, generalización a entradas nuevas, KB desde tuplas, preprocesamiento de un ejemplo inventado" icon="academic-cap" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/key-value-memnn-miller-2016" title="Key-Value MemNN (2016)" subtitle="Miller et al. — memoria (key, value), leer documentos, WikiMovies. El paper que implementa este lab" icon="document-text" >}}
  {{< card link="/papers/e2e-memnn-sukhbaatar-2015" title="End-to-End MemNN (2015)" subtitle="Sukhbaatar et al. — atención softmax sobre memoria, el predecesor directo" icon="document-text" >}}
  {{< card link="/papers/memory-networks-weston-2014" title="Memory Networks (2014)" subtitle="Weston et al. — memoria explícita de slots, el origen de la familia" icon="document-text" >}}
  {{< card link="/papers/babi-weston-2015" title="bAbI (2015)" subtitle="Weston et al. — los benchmarks de razonamiento, de donde sale WikiMovies" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/redes-de-memoria" title="Redes con Memoria Externa" subtitle="Las dos estirpes: Memory Networks (Weston) y memoria diferenciable (Graves)" icon="book-open" >}}
  {{< card link="/fundamentos/self-attention" title="Self-Attention" subtitle="El key addressing + value reading de este lab es exactamente query/key/value" icon="book-open" >}}
  {{< card link="/fundamentos/question-answering" title="Question Answering" subtitle="La tarea que resuelve el lab (WikiMovies)" icon="book-open" >}}
  {{< card link="/fundamentos/dense-retrieval" title="Dense Retrieval" subtitle="El blocking + scoring del lab es un retrieve-then-rank" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-30" title="Clase 30 - Teoría" subtitle="Memoria implícita vs explícita, Memory Networks, E2E MemNN, KV-MemNN, Entity Networks, NTM/DNC" icon="academic-cap" >}}
  {{< card link="/clases/clase-30/profundizacion" title="Profundización" subtitle="Math: lectura por atención suave, MemN2N formal, key-value addressing" icon="beaker" >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Del QA con memory networks a los LLM con RAG" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-27" title="Lab 27 - Redes Neuronales de Grafos (anterior)" subtitle="GNN con PyTorch Geometric" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 82 celdas del notebook + 5 experimentos propios añadidos (búsqueda de errores, top-5 candidatos, top-k accuracy, muestra aleatoria, casos de contraste). Las 4 actividades resueltas, con la Actividad 4 (código) verificada corriendo (`neil marshall` ✓). Notebook ejecutado en Colab con heatmaps de atención embebidos.
