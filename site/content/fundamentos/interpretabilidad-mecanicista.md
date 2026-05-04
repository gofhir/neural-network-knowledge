---
title: "Interpretabilidad mecanicista"
weight: 320
math: true
---

**Interpretabilidad mecanicista** (mech interp) es la disciplina que busca entender que computan los modelos neuronales — no como "cajas negras" de input/output, sino mecanismo por mecanismo: que cabezas implementan que algoritmos, que features representa el residual stream, que circuitos componen el comportamiento. Es una direccion de research desarrollada principalmente por Anthropic (Transformer Circuits Thread) entre 2021 y 2026, con aplicaciones cada vez mas relevantes para alignment y safety. Es el corazon del *Camino 3* del curso.

---

## 1. Apertura: la caja se puede abrir

Hasta 2021, "interpretar" un Transformer significaba mirar attention weights y especular. Patrones eran descriptivos, sin marco matematico riguroso. El paper "A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic 2021) cambio eso: introdujo abstracciones precisas — **residual stream**, **QK circuits**, **OV circuits** — que permiten razonar sobre el modelo como un sistema de computos discretos en lugar de un blob no-diferenciable.

Desde entonces, mech interp produjo descubrimientos concretos:

- **Induction heads** (Olsson et al. 2022): cabezas que implementan in-context learning via patron `[A][B]...[A] -> [B]`
- **Circuito IOI** (Wang et al. 2022): 26 cabezas en GPT-2 small implementando "Indirect Object Identification"
- **Sparse Autoencoders** (Bricken et al. 2023, Templeton et al. 2024): tecnica para descomponer superposition en features monosemanticas

En 2026, mech interp esta en transicion de "research academica" a "herramienta de safety". Anthropic, OpenAI, DeepMind y otros labs invierten en mech interp como mecanismo de validacion para sistemas de IA cada vez mas capaces.

---

## 2. Conceptos centrales

### Residual stream

La "autopista" del Transformer. Cada bloque LEE el residual stream (con atencion + FFN) y ESCRIBE un delta a el (via la conexion residual `x = x + bloque(x)`). Ningun bloque sobreescribe — solo agrega. El head final lee la suma acumulada.

Implicacion: la informacion que el modelo necesita preservar atraviesa todo el modelo via el residual stream. Cada componente tiene un "canal de comunicacion" sobre este bus compartido. Identificar circuitos significa trazar quien escribe que en el stream y quien lo lee.

Ver [cap 51](/clases/clase-14/practica/51-residual-stream).

### Forward hooks

Mecanismo de PyTorch (`register_forward_hook`) para capturar activaciones de cualquier modulo durante un forward pass, sin modificar el codigo del modelo. Es la primitiva que hace posible TODA la interpretabilidad mecanicista practica.

Implementacion estandar: context manager que registra hooks por nombre de modulo, los remueve automaticamente. Ver [`_interp.py:cache_activations`](/clases/clase-14/practica/50-forward-hooks).

### Logit lens

Tecnica (nostalgebraist 2020) que aplica el head del modelo (`lm_head`) al residual stream **intermedio**, no solo al final. Cada capa tiene asi su prediccion provisional del proximo token. Permite trazar como emerge la prediccion capa por capa.

Funciona porque: el head es una matriz lineal `(d_model, vocab)`, y opera sobre vectores de `d_model` dimensiones — los residuales intermedios viven en el mismo espacio que el residual final, asi que el head produce logits validos sobre cualquiera de ellos. Ver [cap 52](/clases/clase-14/practica/52-logit-lens).

### Previous-token heads

Cabezas que sistematicamente atienden al token anterior (`attn[i, i-1]` alto). Score: `previous_token_score(attn) = mean(attn[i, i-1] for i >= 1)`. Anthropic encontro que estas cabezas emergen tipicamente en capas tempranas (0-2) y son la BASE estructural sobre la que se construyen las induction heads. Ver [cap 54](/clases/clase-14/practica/54-previous-token-heads).

### Induction heads

Cabezas que implementan el patron `[A][B] ... [A] -> [B]`: dado un token repetido, atienden al "siguiente token" de la primera ocurrencia. Anthropic (Olsson et al. 2022) las identifico como el sustrato emergente del **in-context learning** — la capacidad de los LLMs de aprender patrones del contexto sin entrenamiento.

Las induction heads emergen tipicamente en capas 2-6 de modelos de tamano mediano (GPT-2 small+). Requieren componer con previous-token heads de capas anteriores. En modelos chicos (Mini-LLaMA con 4 capas) NO emergen claramente. Ver [cap 55](/clases/clase-14/practica/55-induction-heads).

### QK / OV decomposition

Cada cabeza de atencion se descompone matematicamente en dos circuitos:

- **QK circuit** ($W_Q W_K^T$): determina **a que atender** dada una query
- **OV circuit** ($W_V W_O$): determina **que escribir al residual stream** dado un token fuente

Formalizado por Elhage et al. (2021). La descomposicion permite razonar sobre la cabeza de manera modular: el QK responde "donde mira", el OV responde "que mueve". Ver [cap 56](/clases/clase-14/practica/56-qk-ov-decomposition).

### Activation patching

Tecnica canonica para tests causales en redes. Pasos:

1. Correr modelo sobre prompt "clean", cachear activaciones
2. Correr sobre prompt "corrupted" con prediccion distinta
3. Para cada componente, reemplazar la activacion del corrupted con la del clean
4. Medir cuanto cambia la prediccion: el "recovery score"

Si el componente es causalmente importante, patchearlo recupera la prediccion clean. Si no, patchearlo no cambia nada. Es la unica forma de pasar de correlacion (heatmaps, scores descriptivos) a causalidad (efecto real sobre la prediccion). Ver [cap 57](/clases/clase-14/practica/57-activation-patching).

### Superposition

Fenomeno por el cual modelos con `n_features > d_model` aprenden representaciones NO-ortogonales — comprimiendo features en angulos distintos del espacio. Consecuencia: las **neuronas son polisemanticas** (cada una responde a multiples conceptos no relacionados).

Formalizado por Anthropic (Elhage et al. 2022, "Toy Models of Superposition"). La interpretabilidad neuron-by-neuron NO funciona en modelos grandes precisamente por superposition — necesitas SAEs para descomponer el espacio en features monosemanticas. Ver [cap 59](/clases/clase-14/practica/59-superposition).

### Sparse Autoencoders (SAEs)

Arquitectura para deshacer superposition:

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_features, l1_coeff):
        # encoder: Linear(d_model, d_features) + ReLU
        # decoder: Linear(d_features, d_model, bias=False)
    
    def forward(self, x):
        features = relu(self.encoder(x))
        recon = self.decoder(features)
        return recon, features

# loss = MSE(reconstruction) + l1_coeff * L1(features)
```

Re-representan el residual stream en un espacio MAS GRANDE (`d_features > d_model`) pero con activacion ESPARSA. La sparsity penalty obliga a que solo unas pocas features esten activas por input — bajo esta condicion, las features tienden a ser monosemanticas (representan UN concepto cada una).

Anthropic (Templeton et al. 2024, "Scaling Monosemanticity") escalo SAEs a Claude 3 Sonnet con `d_features` de 34 millones, descubriendo features para conceptos abstractos como "Golden Gate Bridge", "inner conflict", "code vulnerabilities". Ver [cap 60-61](/clases/clase-14/practica/60-train-sae).

### Circuits

Subgrafos del modelo (conjuntos de cabezas + MLPs + features) que implementan algoritmos especificos. Identificados via composicion: descomposicion QK/OV (cap 56) + activation patching (cap 57-58) + interpretacion de features SAE (cap 61).

El "santo grial" de mech interp. Ejemplos canonicos:

- **Circuito IOI** (Wang et al. 2022): 26 cabezas en GPT-2 small resolviendo "John gave Mary a flower. Mary gave a flower to ___"
- **Modular addition** (Nanda et al. 2023): circuito en transformers entrenados para hacer aritmetica modular
- **Indirect speech acts**: features y cabezas que codifican intencion comunicativa

---

## 3. Lecciones del Camino 3

### Descripcion ≠ causalidad

La cabeza con MAYOR previous-token score en Mini-LLaMA (block.2 head.0, score 0.547) tiene recovery causal NEGATIVO (-2.7%) cuando se patchea (cap 58). Las cabezas con scores prev-token bajos resultan ser las MAS causales para distinguir speakers.

Implicacion: las metricas descriptivas son utiles para identificar candidatos, NO conclusivas para identificar funcion. Necesitas patching para validar.

### Escala importa

Mini-LLaMA (4 capas, 890K params) NO produce induction heads claras (cap 55). Anthropic las encontro en GPT-2 small (12 capas, 115M params). La mech interp transfiere las TECNICAS pero algunos PATRONES requieren escala.

### Superposition es universal pero deshacible

Cap 59 mostro superposition en toy model. Cap 60-61 entreno un SAE sobre Mini-LLaMA y descubrio que **47% de las features son monosemanticas**. La superposition es real, pero los SAEs son herramientas efectivas para deshacerla en cualquier escala.

---

## 4. Aplicaciones en 2026

### Sparse Autoencoders a escala

Anthropic Scaling Monosemanticity (2024) entreno SAEs sobre Claude 3 Sonnet con `d_features` ~34M. Descubrieron features causalmente activas: activar artificialmente "Golden Gate Bridge feature" hace que el modelo hable obsesivamente del Golden Gate Bridge en cualquier contexto. Es la "intervencion" mas cercana a "leer la mente" del modelo.

### Mech interp para alignment

El programa de Anthropic y otros labs es usar mech interp para:

1. **Detectar capacidades peligrosas** antes de deployment (sleeper agents, deception)
2. **Validar comportamientos seguros** mecanisticamente, no solo via benchmarks
3. **Disenar intervenciones** sobre features causales para suprimir comportamientos no deseados

Es una linea de research donde mech interp pasa de academic a safety-critical.

### TransformerLens y herramientas profesionales

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (Neel Nanda) es la libreria estandar para investigacion mech interp. Provee:

- API para cargar modelos pre-entrenados (GPT-2, Pythia, LLaMA) con hooks pre-instrumentados
- Implementaciones de activation patching, ablation, attribution patching
- Toy models para estudiar fenomenos especificos
- Tutorials reproducibles

Otras: nnsight (NDIF), inseq, circuitsvis, SAELens.

---

## 5. Lugar en el curso

Camino 3 (caps 50-63) implementa mech interp DESDE CERO sobre Mini-LLaMA y Mini-BERT — fiel a la pedagogia "you build it, you understand it".

- **Caps 50-52**: hooks, residual stream, logit lens
- **Caps 53-56**: heatmaps de atencion, previous-token, induction (no emerge), QK/OV
- **Caps 57-58**: activation patching, head-level patching
- **Caps 59-61**: superposition, sparse autoencoders, interpretacion de features
- **Cap 62**: contraste con Mini-BERT (encoder bidireccional)
- **Cap 63**: comparativa final + frontera 2026

Caminos relacionados:

- [Mecanismo de atencion](/fundamentos/mecanismo-atencion) — base teorica que mech interp diseca
- [Foundation Models](/fundamentos/foundation-models) — los modelos sobre los que se aplica mech interp
- [BERT](/fundamentos/bert) — encoder-only, contrastado en cap 62

El campo de mech interp esta activo y avanza rapido. Camino 3 da la base; las herramientas profesionales (TransformerLens, SAELens) son el siguiente paso para investigacion seria.

---

## 6. Referencias claves

- **Elhage, N., Nanda, N., et al. (2021).** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/). Anthropic.
- **Olsson, C., Elhage, N., et al. (2022).** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.
- **Elhage, N., Hume, T., et al. (2022).** [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/). Anthropic.
- **Wang, K., Variengien, A., et al. (2022).** [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593). Redwood Research.
- **Bricken, T., Templeton, A., et al. (2023).** [Towards Monosemanticity: Decomposing Language Models with Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/). Anthropic.
- **Templeton, A., Conerly, T., et al. (2024).** [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/). Anthropic.
- **TransformerLens** (Neel Nanda): https://github.com/TransformerLensOrg/TransformerLens
- **ARENA**: https://arena3-chapter1-transformer-interp.streamlit.app/
- **Anthropic Circuits Thread**: https://transformer-circuits.pub/

Mech interp es uno de los pocos campos en ML donde el progreso es comprensible — no es "magia" como muchos otros aspectos del deep learning. Cada experimento se basa en algebra lineal y observacion cuidadosa. Es lento pero satisfactorio.
