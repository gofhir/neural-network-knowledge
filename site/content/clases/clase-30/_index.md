---
title: "Clase 30 - Modelos con memoria externa"
weight: 300
sidebar:
  open: true
---

**Profesor:** Andrés Espinosa
**Curso 3 / Tópicos de profundización:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase sobre **redes con memoria externa**: arquitecturas que guardan información en una **memoria explícita** de slots —interpretable y editable después del entrenamiento— en lugar de dejarla atrapada implícitamente en los pesos. La motivación es clara: en una red tradicional es difícil agregar conocimiento nuevo sin borrar el anterior, y la memoria es poco interpretable. La clase recorre la **línea de Memory Networks** de Weston para razonamiento y QA —**Memory Networks → End-to-End MemNN → Key-Value MemNN → Recurrent Entity Networks**— sobre los datasets **bAbI** y **WikiMovies**, y conecta con la otra gran estirpe (la **memoria diferenciable tipo computador**: NTM y DNC).

La clase es, en el fondo, la prehistoria de dos ideas centrales del deep learning moderno: la **self-attention** de los Transformers (que es, literalmente, lectura de una memoria por atención suave) y el **RAG** (retrieval-augmented generation, memoria externa para los LLM). Se apoya en el [mecanismo de atención (Clase 15)](/clases/clase-15) y conecta con el [meta-aprendizaje (Clase 26)](/clases/clase-26), donde MANN usa memoria externa para one-shot.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 21 diapositivas: memoria implícita vs explícita, bAbI, Memory Networks, End-to-End MemNN, Key-Value MemNN, Entity Networks" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: lectura por atención suave, MemN2N formal, key-value addressing, celda de EntNet, NTM (content/location), DNC" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="End-to-End MemNN y memoria diferenciable (NTM) desde cero en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-30" title="Laboratorio: Key-Value Memory Networks" subtitle="Implementar KV-MemNN sobre WikiMovies QA, con 5 experimentos propios y las 4 actividades resueltas" icon="beaker" >}}
  {{< card link="/clases/clase-31" title="Clase siguiente: Aprendizaje Reforzado" subtitle="Q-Learning, DQN, policy gradient" icon="arrow-right" >}}
  {{< card link="/clases/clase-29" title="Clase anterior: Modelos Generativos en Visión" subtitle="VAE, GAN, difusión, Stable Diffusion" icon="arrow-left" >}}
  {{< card link="/clases/clase-15" title="Base: Mecanismo de atención" subtitle="La lectura de memoria por atención suave" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/redes-de-memoria" title="Redes con Memoria Externa" subtitle="Las dos estirpes: Memory Networks (Weston) y memoria diferenciable (Graves)" icon="book-open" >}}
  {{< card link="/fundamentos/memory-augmented-networks" title="Memory-Augmented Networks" subtitle="NTM, MANN/LRUA, acceso por contenido, conexión con la atención" icon="book-open" >}}
  {{< card link="/fundamentos/self-attention" title="Self-Attention" subtitle="La self-attention de los Transformers es lectura de memoria (query/key/value)" icon="book-open" >}}
  {{< card link="/fundamentos/question-answering" title="Question Answering" subtitle="La tarea que las memory networks resuelven (bAbI, WikiMovies)" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/memory-networks-weston-2014" title="Memory Networks (2014)" subtitle="Weston et al. — memoria explícita de slots, componentes I/G/O/R" icon="document-text" >}}
  {{< card link="/papers/e2e-memnn-sukhbaatar-2015" title="End-to-End MemNN (2015)" subtitle="Sukhbaatar et al. — atención softmax, sin supervisión de hops, precursor de la self-attention" icon="document-text" >}}
  {{< card link="/papers/key-value-memnn-miller-2016" title="Key-Value MemNN (2016)" subtitle="Miller et al. — memoria (key, value), leer documentos, WikiMovies" icon="document-text" >}}
  {{< card link="/papers/entity-networks-henaff-2017" title="Recurrent Entity Networks (2017)" subtitle="Henaff et al. — un slot por entidad, world state, primer modelo en pasar bAbI" icon="document-text" >}}
  {{< card link="/papers/babi-weston-2015" title="bAbI (2015)" subtitle="Weston et al. — las 20 tareas de razonamiento del campo" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/ntm-graves-2014" title="Neural Turing Machines (2014)" subtitle="Graves et al. — memoria externa diferenciable, aprende algoritmos" icon="document-text" >}}
  {{< card link="/papers/dnc-graves-2016" title="Differentiable Neural Computer (2016)" subtitle="Graves et al. (Nature) — asignación dinámica + enlaces temporales, razonamiento sobre grafos" icon="document-text" >}}
  {{< card link="/papers/mann-santoro-2016" title="MANN (2016)" subtitle="Santoro et al. — memoria externa para meta-aprendizaje one-shot" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/texto" title="Dominio: Texto / NLP" subtitle="Línea de tiempo: del QA con memory networks a los LLM con RAG" icon="globe-alt" >}}
{{< /cards >}}
