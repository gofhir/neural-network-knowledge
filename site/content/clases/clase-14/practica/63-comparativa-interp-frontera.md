---
title: "63 - Comparativa interp + frontera 2026 — cierre Camino 3"
weight: 630
math: true
---

## 1. Apertura

Catorce capitulos despues del cap 50, el Camino 3 cierra. El residual stream existe, los hooks funcionan, los heatmaps de atencion son legibles, las cabezas previous-token estan identificadas, los SAEs entrenados, las features inspeccionadas, y BERT comparado con LLaMA. ¿Que aprendimos? ¿Donde estamos en 2026 vs el frontier de research?

Este capitulo no agrega codigo. Es una sintesis de las tecnicas, una comparativa con la frontera moderna (Anthropic Circuits Thread, Sparse Autoencoders a escala, mech interp para alignment), y un cierre que conecta Camino 3 con los caminos pendientes (Camino 5: ViT) y con la realidad de la investigacion industrial.

---

## 2. Tabla maestra de tecnicas del Camino 3

| Tecnica | Cap | Que hace | Resultado en Mini-LLaMA |
|---|---|---|---|
| Forward hooks | 50 | Cachear activaciones de cualquier modulo sin modificar el modelo | norm bloque 3 explota a 25.79 (vs 9-12 en otros) |
| Residual stream | 51 | Visualizar la "autopista" de informacion | Bloque 3 escribe ||delta||/||in||=1.64 |
| Logit lens | 52 | Predicciones capa por capa via head | El modelo NO predice 'b' tras "To be or not to " (limitacion escala) |
| Heatmaps de atencion | 53 | Visualizar patrones (T, T) por cabeza | 16 cabezas con patrones distintos identificables |
| Previous-token heads | 54 | Score sobre 50 prompts | block.2 head.0 con 0.547 (top-1) |
| Induction heads | 55 | Score sobre prompts repetidos | NO emergen (top score 0.057) — escala insuficiente |
| QK / OV decomposition | 56 | Descomposicion matematica de cabezas | Top prev-token NO es copy head (||OV-I||/||I||=1.04) |
| Activation patching | 57 | Intervencion causal posicion-por-posicion | Flujo causal del speaker hacia posicion 12 |
| Head-level patching | 58 | Intervencion causal cabeza-por-cabeza | 4 cabezas explican circuito speaker. Top prev-token tiene recovery NEGATIVO |
| Superposition (toy) | 59 | Demostrar polisemanticidad | Cluster + anti-pareo emerge con 5 features en 2D |
| Train SAE | 60 | Sparse autoencoder sobre block.2 | 98.4% var explicada, L0=166/512 |
| Interpret SAE | 61 | Top-k tokens por feature | 47% monosemanticas (242/512) |
| Mini-BERT contrast | 62 | Aplicar tecnicas a encoder | Capa 3 distingue EN/ES via [SEP] vs [CLS] aggregation |

---

## 3. Las tres lecciones centrales del Camino 3

### Leccion 1: descripcion ≠ causalidad

La cabeza con MAYOR previous-token score (block.2 head.0, score 0.547 en cap 54) tiene recovery causal NEGATIVO (-2.7%) cuando se patchea para distinguir speakers (cap 58). Las cabezas con scores prev-token bajos (block.0 head.1 con score 0.144, EL MAS BAJO) son las MAS causales (+24.7%).

**Implicacion**: las metricas descriptivas (heatmaps, patrones, scores) son utiles para identificar candidatos, NO conclusivas para identificar funcion. La interpretabilidad mecanicista requiere validacion causal via patching.

Anthropic (Wang et al. 2022) llego a la misma conclusion en GPT-2 small con el circuito IOI: cabezas que el "prior intuitivo" sugeria importantes resultaron ser secundarias; cabezas no obvias resultaron centrales. La leccion se generaliza.

### Leccion 2: escala importa para que algunos patrones emerjan

Cap 55 mostro que las induction heads NO emergen en Mini-LLaMA (4 capas, d_model=128). Anthropic las encontro en GPT-2 small (12 capas, d_model=768, ~115M params). La escala es ~130x mayor en parametros.

Esto no invalida la teoria — confirma que las induction heads son emergentes a escala adecuada. Los modelos chicos pueden estudiarse con las mismas tecnicas, pero algunos patrones especificos requieren MAS profundidad o MAS dimensiones.

Implicacion practica: para investigacion de research-grade en mech interp, los modelos minimos son ~3 capas / d_model 64+ con tareas sinteticas controladas (los "attn-only-2L" de TransformerLens). Mini-LLaMA esta en el limite inferior — permite mostrar el toolkit, no necesariamente reproducir todos los descubrimientos.

### Leccion 3: superposition es una propiedad universal, deshacerla es factible

Cap 59 mostro superposition en un toy model. Cap 60-61 entreno un SAE sobre Mini-LLaMA y descubrio que **47% de las features son monosemanticas** — caracteres especificos, signos de puntuacion, separadores estructurales.

Esto es notable: con minimo tuning (L1=0.5, 2000 iters, d_features=512), el SAE descubre estructura interpretable. La frontera 2024-2026 (Anthropic Scaling Monosemanticity) escalas esto a millones de features sobre LLMs grandes con resultados aun mas claros — pero el principio se valida ya en modelos chicos.

---

## 4. La frontera 2024-2026

### Sparse Autoencoders a escala

El paper "Scaling Monosemanticity" (Templeton et al. 2024) entreno SAEs sobre Claude 3 Sonnet con `d_features` del orden de **34 millones**. Encontraron features para:

- Conceptos abstractos: "Golden Gate Bridge", "inner conflict", "code vulnerabilities"
- Operaciones: "negation", "comparison", "arithmetic"
- Idiomas: features especificas para cada idioma soportado
- Caracteristicas de personajes: politeness, sarcasm, assertiveness

Lo notable: las features son **causalmente activas**. Si activas artificialmente la feature "Golden Gate Bridge" en el residual stream, el modelo empieza a hablar OBSESIVAMENTE del Golden Gate Bridge en cualquier contexto. Esto demostro que las features no son solo descriptivas — son los building blocks reales del comportamiento del modelo.

### Activation Patching como cirugia

DeepMind y Anthropic han desarrollado tecnicas mas finas que el patching basico de cap 57:

- **Path patching**: patchear conexiones especificas entre componentes (cabeza A → cabeza B), no componentes individuales. Permite aislar circuitos en grafos complejos.
- **Edge patching**: patchear las contribuciones que UN componente le envia a OTRO especifico. Diseccion mas fina que path patching.
- **Attribution patching**: aproximaciones lineales que permiten patchear MILES de componentes en una sola corrida (vs cientos con patching directo).

Estas tecnicas son el state-of-the-art 2025-2026. Para Mini-LLaMA seria over-engineering; para investigaciones reales sobre circuitos en GPT-2/Pythia/LLaMA-7B son esenciales.

### Mech interp para alignment

El programa de Anthropic (y otros labs) es usar mech interp para:

1. **Detectar capacidades peligrosas** antes de deployment (sleeper agents, deception)
2. **Validar comportamientos seguros** mecanisticamente, no solo via benchmarks
3. **Diseñar intervenciones** sobre features causales para suprimir comportamientos no deseados

Es una linea de research activa donde mech interp deja de ser academica y se vuelve safety-critical. Camino 3 es el "intro 101" de un campo que esta en pleno desarrollo.

---

## 5. TransformerLens y el ecosistema profesional

Camino 3 fue construido enteramente desde cero — fiel a la filosofia "you build it, you understand it" del curso. Pero para investigacion seria, la herramienta estandar es **TransformerLens** (Neel Nanda).

TransformerLens provee:

- API para cargar modelos pre-entrenados (GPT-2, Pythia, LLaMA, etc.) con hooks pre-instrumentados
- Implementaciones eficientes de activation patching, ablation, etc.
- Modelos toy especificamente diseñados para estudiar fenomenos de mech interp (`solu-2l`, `attn-only-2l`)
- Tutorials y ejemplos de circuitos reproducidos

Si hubieramos usado TransformerLens en Camino 3, hubiera sido 5x mas rapido — pero menos pedagogico. Saber implementar `cache_activations` con `register_forward_hook` da una base que ninguna libreria puede dar. Para investigaciones futuras, TransformerLens es el siguiente paso natural.

Otras herramientas relevantes:

- **nnsight** (NDIF): biblioteca para experimentos mech interp distribuidos
- **inseq**: framework para attribution patching
- **circuitsvis**: visualizacion de patrones de atencion en notebooks
- **SAELens**: API para entrenar y usar Sparse Autoencoders

---

## 6. Lo que descubrimos sobre Mini-LLaMA

Resumen de hallazgos sobre el modelo del curso:

**Estructura del residual stream:**

- Bloque 0: edits suaves (delta/in=0.48), construye features posicionales
- Bloque 1: transicion, primer previous-token head emerge
- Bloque 2: previous-token heads dominantes (top score 0.547)
- Bloque 3: explosion de magnitud (delta/in=1.64), cristalizacion de prediccion

**Cabezas relevantes** (cap 58, head-level patching):

- block.0 head.1: causal +24.7% (NO previous-token)
- block.0 head.3: causal +29.7% (mixed pattern)
- block.1 head.2: causal +29.7% (previous-token con prev_score=0.467)
- block.2 head.3: causal +23.5%
- block.2 head.0: descriptivamente top prev-token, causalmente NEGATIVA

**Sparse autoencoder sobre block.2** (cap 60-61):

- 512 features, 47% monosemanticas
- Features descubiertas: caracteres individuales (e, o, s, h, r, v), puntuacion (`.`, `,`, `\n`, `?`, `:`)
- Reconstruccion 98.4%, L0=166

**Comparacion con Mini-BERT** (cap 62):

- BERT atiende [CLS]/[SEP] como mecanismo de pooling
- Capa 3 de BERT distingue idiomas: EN deposita en [SEP] (0.377), ES en [CLS] (0.101)
- Vectores [CLS] EN vs ES son casi ortogonales (cosine 0.002) — explica accuracy 99.8%

---

## 7. Preguntas finales del Camino 3

**1. ¿Que tarea pedirias a Mini-LLaMA donde la interpretabilidad mecanicista te ayudaria a resolver un problema concreto?**

Ejemplos de tareas para las que la interpretabilidad seria informativa:

- **Mode collapse en fine-tuning**: si DPO degrada accuracy (como vimos en cap 29), ¿cuales cabezas cambiaron? Patcheo entre el SFT y el DPO checkpoint identificaria los componentes que se "rompieron".
- **Eleccion entre prompts**: dado un prompt ambiguo donde el modelo predice algo extrano, ¿que feature en el SAE se activo? Identificarla daria insight sobre el "razonamiento" del modelo.
- **Comparacion entre seeds**: dos modelos entrenados con seeds distintos pero misma arquitectura aprenden cosas similares? El SAE de cada uno tendra features comparables o sera completamente distinto?

Estas preguntas son investigaciones reales que se pueden hacer con el toolkit de Camino 3.

**2. ¿Por que la interpretabilidad mecanicista es relevante para safety y no solo para curiosidad cientifica?**

Si entrenas un LLM grande, las metricas de evaluacion (benchmarks, accuracy) te dicen QUE hace el modelo en promedio, pero no COMO. Un modelo puede pasar todos los benchmarks de safety mientras esconde un mecanismo interno para comportamientos peligrosos en contextos especificos (sleeper agents). Sin mech interp, no hay forma sistematica de detectar esto. Con mech interp, podrias:

- Identificar features especificas que se activan en contextos sensibles (e.g., "feature de evasion de filtros")
- Validar que la cadena causal de razonamiento del modelo es honest (no dice una cosa por fuera y procesa otra por dentro)
- Intervenir activamente para suprimir comportamientos no deseados sin re-entrenar

Esto es activamente trabajado por los teams de safety en Anthropic, OpenAI, DeepMind. Mech interp pasa de academic a producto.

**3. Si tuvieras que recomendar a alguien EMPEZAR a aprender mech interp en 2026, ¿que les dirias?**

Un curso practico:

1. Implementa los basicos desde cero (este Camino 3) sobre un modelo chiquito. Te da intuicion mecanicista.
2. Lee "A Mathematical Framework for Transformer Circuits" (Anthropic 2021). Es la base teorica.
3. Trabaja a traves de "Interpretability in the Wild: IOI in GPT-2" (Wang et al. 2022). Reproduce el circuito.
4. Aprende TransformerLens y ARENA. ARENA es un curso intensivo gratis de Neel Nanda con ejercicios reproducibles.
5. Sigue el Anthropic Circuits Thread mensualmente. La frontera cambia rapido.
6. Lee papers recientes de Sparse Autoencoders. La direccion futura.
7. Une un Reading group o comunidad (LessWrong AF, Anthropic Discord, ARENA students). El campo es chico y conversacional.

El time-to-publish-research es ~6-12 meses si vienes con background en ML. Para alguien empezando desde cero: ~1-2 anos a investigador independiente.

---

## 8. Cierre del Camino 3

Mini-LLaMA y Mini-BERT viven ahora bajo una luz distinta. No son cajas opacas: son arquitecturas con residual streams, cabezas con patrones identificables, circuitos parcialmente trazables, y features descomponibles en sparse autoencoders.

El toolkit del Camino 3 (`_interp.py` con sus 7 helpers + `SparseAutoencoder`) es transferible. Las tecnicas funcionan en LLaMA-7B, Claude, GPT-4 — solo cambia la escala. Lo que aprendiste aqui es la base de todo lo que se hace en mech interp en 2026.

**Caminos pendientes:**

- **Camino 5: Vision Transformer (ViT)** — el encoder BERT aplicado a parches de imagen. Mismas tecnicas de interpretabilidad, distinta modalidad.

Para retomar: el toolkit de `_interp.py` ya esta listo, los modelos entrenados, los hooks funcionando. La frontera de mech interp en 2026 es:

1. Sparse Autoencoders a escala (decenas de millones de features)
2. Mech interp como herramienta de alignment (safety mechanistic verification)
3. Cross-model analysis (¿como cambian los circuitos entre versiones de un mismo modelo?)
4. Multimodal mech interp (CLIP, ViT, modelos vision-language)

Una nota personal: la interpretabilidad mecanicista es uno de los pocos campos en ML donde el progreso es comprensible — no es "magia" como muchos otros aspectos del deep learning. Cada experimento se basa en algebra lineal y observacion cuidadosa. Es lento pero satisfactorio. Si te gusto Camino 3, este es un campo para profundizar.

---

## 9. Referencias claves

- **Elhage, N., Nanda, N., et al. (2021).** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). Anthropic.
- **Olsson, C., Elhage, N., et al. (2022).** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.
- **Elhage, N., Hume, T., et al. (2022).** [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/). Anthropic.
- **Wang, K., Variengien, A., et al. (2022).** [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593). Redwood Research.
- **Bricken, T., Templeton, A., et al. (2023).** [Towards Monosemanticity: Decomposing Language Models with Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/). Anthropic.
- **Templeton, A., Conerly, T., et al. (2024).** [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/). Anthropic.
- **TransformerLens** (Neel Nanda): https://github.com/TransformerLensOrg/TransformerLens
- **ARENA**: https://arena3-chapter1-transformer-interp.streamlit.app/
- **Anthropic Circuits Thread**: https://transformer-circuits.pub/

Camino 3 cerrado. Mini-LLaMA y Mini-BERT bajo el microscopio. La frontera 2026 abierta.
