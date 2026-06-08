---
title: "Practica desde 0 - Meta-aprendizaje"
weight: 30
sidebar:
  open: true
---

La clase 26 cubre el meta-aprendizaje: aprender *a aprender* para adaptarse a tareas nuevas con pocos ejemplos. Esta practica implementa sus algoritmos emblematicos en **minima escala** para entender por dentro que los distingue, no solo leer sus papers. Partimos de la infraestructura comun (el sampler episodico N-way K-shot), seguimos con los dos metodos metric-based mas didacticos (Prototypical y Matching Networks), construimos MAML con su diferenciacion de segundo orden, y cerramos con redes siamesas conectadas a un problema real de salud: el *record linkage* / patient matching. Cuando aplica, replicamos el mismo modelo en **triple framework** (PyTorch, TensorFlow y JAX/Flax) para ver como cada uno expresa las mismas ideas — y por que JAX brilla en la diferenciacion anidada de MAML.

## Caminos

{{< cards >}}
  {{< card link="01-episodios-nway-kshot" title="01 - Episodios N-way K-shot" subtitle="El sampler episodico (support/query) desde 0 en PyTorch — la base de todo lo demas" icon="code" >}}
  {{< card link="02-prototypical-net" title="02 - Prototypical Networks" subtitle="Prototipos + softmax sobre distancias en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="03-maml" title="03 - MAML desde 0" subtitle="Inner/outer loop y segundo orden en PyTorch, TensorFlow y JAX (regresion sinusoidal)" icon="code" >}}
  {{< card link="04-siamese-verificacion" title="04 - Redes Siamesas y verificacion" subtitle="Metric learning por pares en PyTorch + conexion con record linkage / patient matching" icon="code" >}}
  {{< card link="05-matching-networks" title="05 - Matching Networks" subtitle="Atencion sobre el support set en PyTorch — el puente hacia los Transformers" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 13 - Transfer learning y fine-tuning](../../clase-13): la idea de reutilizar conocimiento previo, que MAML lleva al extremo de "inicializacion adaptable".
- [Clase 14 - Transformer desde 0](../../clase-14/practica): self-attention key-value — Matching Networks y MANN son atencion sobre una coleccion, asi que ayuda haberla visto.
- Python intermedio (clases, decoradores) y NumPy.
- PyTorch basico (tensores, `nn.Module`, autograd, training loop). Util pero no obligatorio: nociones de TensorFlow/Keras y JAX/Flax — los caminos 02 y 03 los introducen comparativamente.
- GPU **recomendada** pero no obligatoria. Todos los caminos corren en CPU (los toy datasets son pequeños); con GPU bajan a segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - Episodios N-way K-shot | PyTorch 2.x | — |
| 02 - Prototypical Networks | PyTorch 2.x | TensorFlow 2.x, JAX + Flax |
| 03 - MAML desde 0 | PyTorch 2.x | TensorFlow 2.x, JAX + Flax |
| 04 - Redes Siamesas | PyTorch 2.x | — |
| 05 - Matching Networks | PyTorch 2.x | — |

Versiones de referencia: `torch>=2.2`, `tensorflow>=2.15`, `jax>=0.4` con `flax>=0.8`, `optax>=0.2`, `torchvision>=0.17` (opcional, para Omniglot real).

## Estructura comun de los caminos

Cada camino sigue el mismo arco pedagogico:

1. **Motivacion**: que problema de few-shot resuelve este metodo y como.
2. **Setup**: dependencias, dataset episodico (sintetico que siempre corre, u Omniglot si esta disponible).
3. **Implementacion paso a paso**: cada componente (encoder, kernel/distancia, loop episodico) con shapes anotadas.
4. **Entrenamiento mini**: pocos episodios para ver la loss bajar y validar que el codigo "aprende a aprender".
5. **Evaluacion**: accuracy N-way K-shot sobre episodios de test, con **intervalos de confianza al 95%** (la metrica estandar del campo).
6. **Discusion**: limitaciones de la version mini, que cambia a escala, lectura honesta de los resultados.
7. **Siguientes pasos**: papers, escalado, y conexiones con otros caminos.

---

**Ver tambien:** [Clase 26 - Teoria](../teoria) · [Clase 26 - Profundizacion](../profundizacion) · [Clase 14 - Transformer desde 0](../../clase-14/practica) · Fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) · [Few-shot Learning](/fundamentos/few-shot-learning).
