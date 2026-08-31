---
title: "XLNet: Generalized Autoregressive Pretraining (2019)"
weight: 296
math: true
---

{{< paper-card
    title="XLNet: Generalized Autoregressive Pretraining for Language Understanding"
    authors="Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le (CMU / Google Brain)"
    year="2019"
    venue="NeurIPS 2019 / arXiv:1906.08237"
    arxiv="1906.08237"
    pdf="/papers/xlnet-yang-2019.pdf" >}}
El primer modelo que superó a [BERT](/papers/bert-devlin-2018) en las veinte tareas del benchmark, atacando un defecto concreto de su objetivo de entrenamiento: al enmascarar varios tokens a la vez, BERT los predice **de forma independiente** y no puede modelar la dependencia entre ellos. XLNet conserva el contexto bidireccional pero recupera la factorización autoregresiva, permutando el **orden de factorización** en lugar del orden de la secuencia. El precio es una arquitectura de **dos flujos de atención** que existe solo para resolver un problema que la idea misma crea. Aparece en el [Laboratorio 20](/laboratorios/lab-20).
{{< /paper-card >}}

---

## El defecto que ataca

BERT enmascara ~15 % de los tokens y los predice en paralelo. Sobre la oración *"New York is a city"*, eligiendo `New` y `York` como objetivos:

$$\mathcal{L}_{\text{BERT}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \text{is a city})$$

$$\mathcal{L}_{\text{XLNet}} = \log p(\text{New} \mid \text{is a city}) + \log p(\text{York} \mid \mathbf{New},\ \text{is a city})$$

Para predecir `York`, XLNet **condiciona en `New`**. BERT no puede: ambos están enmascarados a la vez y se predicen por separado, así que la dependencia entre las dos mitades de un nombre propio nunca entra en la loss.

El paper formaliza la cobertura (apéndice A.5.1): para un par objetivo-contexto $(x, \mathcal{U})$, BERT cubre la dependencia solo si $\mathcal{U} \subseteq \mathcal{N}$ (los no-objetivos), mientras XLNet la cubre si $\mathcal{U} \subseteq \mathcal{N} \cup \mathcal{T}_{<x}$ — los no-objetivos **más los objetivos anteriores en el orden de factorización**. Estrictamente más.

Hay un segundo defecto, de otra naturaleza: el token `[MASK]` aparece en pre-entrenamiento y **nunca** en fine-tuning. XLNet no lo usa.

## Permutation Language Modeling

Sea $\mathcal{Z}_T$ el conjunto de las $T!$ permutaciones de los índices. El objetivo (ec. 3):

$$\max_{\theta} \quad \mathbb{E}_{z \sim \mathcal{Z}_T}\left[\sum_{t=1}^{T} \log p_\theta(x_{z_t} \mid x_{z_{<t}})\right]$$

Como los parámetros se comparten entre todas las permutaciones, **en expectativa cada token ve a todos los demás como contexto** — se recupera la bidireccionalidad sin enmascarar.

{{< concept-alert type="clave" >}}
**Se permuta el orden de factorización, no la secuencia.** El paper lo enfatiza en §2.2 porque es el malentendido natural: las palabras siguen en su orden original. Lo que cambia es **qué tokens están ocultos en cada paso de predicción y en qué orden se revelan**.

La razón es práctica: en fine-tuning el modelo verá texto en orden natural, así que entrenar sobre secuencias barajadas produciría un desajuste. Y la implementación lo deja claro — **la permutación se logra manipulando la máscara de atención**, no reordenando los embeddings.
{{< /concept-alert >}}

**Predicción parcial.** Predecir los $T$ tokens de cada orden converge muy lento, porque los primeros en el orden casi no tienen contexto. La solución (§2.3) es predecir solo el último $1/K$ de cada permutación. XLNet-Large usa $K = 6$: **el último ~16 %**. La cercanía con el 15 % de BERT no es casual — ambos buscan el mismo equilibrio entre señal de entrenamiento y costo.

## Two-Stream Self-Attention: el problema que la idea crea

La idea de PLM es simple; implementarla con un Transformer estándar **no funciona**, y el apéndice A.1 lo demuestra.

Con la parametrización habitual $p(X_{z_t} = x \mid x_{z_{<t}}) \propto \exp(e(x)^\top h_\theta(x_{z_{<t}}))$, dos permutaciones que comparten prefijo pero difieren en la posición objetivo ($z^{(1)}_t = i \ne j = z^{(2)}_t$) producen **exactamente la misma predicción**:

$$p(X_i = x \mid x_{z_{<t}}) = p(X_j = x \mid x_{z_{<t}})$$

Es absurdo: el token correcto en la posición $i$ normalmente no es el de la posición $j$. El estado oculto sabe *qué* hay en el contexto pero no *dónde* está prediciendo.

La reparametrización (ec. 4) hace que la representación tome la posición objetivo como entrada:

$$p_\theta(X_{z_t} = x \mid x_{z_{<t}}) = \frac{\exp\big(e(x)^\top g_\theta(x_{z_{<t}}, z_t)\big)}{\sum_{x'} \exp\big(e(x')^\top g_\theta(x_{z_{<t}}, z_t)\big)}$$

Y ahí aparece el conflicto operativo, con dos requisitos incompatibles en un solo estado oculto:

1. Para predecir $x_{z_t}$, la representación necesita la **posición** $z_t$ pero **no el contenido** — o el modelo aprende la identidad.
2. Para predecir tokens posteriores, la representación **sí necesita el contenido** $x_{z_t}$, o la cadena autoregresiva se corta.

**La solución: dos flujos en paralelo**, con parámetros compartidos y máscaras distintas.

$$g_{z_t}^{(m)} \leftarrow \text{Attention}\big(Q = g_{z_t}^{(m-1)},\ KV = h_{z_{<t}}^{(m-1)}\big) \qquad \text{(query: usa } z_t,\ \text{no ve } x_{z_t})$$

$$h_{z_t}^{(m)} \leftarrow \text{Attention}\big(Q = h_{z_t}^{(m-1)},\ KV = h_{z_{\le t}}^{(m-1)}\big) \qquad \text{(content: usa ambos)}$$

La única diferencia es el rango de claves y valores: $z_{<t}$ contra $z_{\le t}$. No son dos modelos — son dos vistas del mismo con máscaras distintas.

Y un detalle que cambia el costo de producción: **en fine-tuning solo se usa el content stream**. El query stream se descarta, así que el modelo servido se comporta como un Transformer-XL normal. Toda la maquinaria de dos flujos existe únicamente durante el pre-entrenamiento.

## Lo que hereda de Transformer-XL

**Recurrencia por segmentos.** Los estados ocultos del segmento anterior se cachean y se inyectan como claves y valores adicionales, dando contexto efectivo de $2T$ sin pagar el costo cuadrático.

El detalle fino (§2.4): **la caché es agnóstica a la permutación**. Se computó bajo alguna permutación del segmento previo, pero como las posiciones son relativas el segmento actual puede reusarla sin saber cuál fue. Es lo que hace el sistema entrenable a escala sin llevar registro de permutaciones en el pipeline de datos.

**Codificación posicional relativa.** Con posiciones absolutas, permutar rompería el significado aprendido de cada índice, y la recurrencia produciría colisiones entre segmentos. La atención relativa resuelve ambos:

$$A_{ij}^{\text{rel}} = \underbrace{q_i^\top W^q W^k_{e} k_j}_{\text{contenido-contenido}} + \underbrace{q_i^\top W^q W^k_{r} R_{i-j}}_{\text{contenido-posición}} + \underbrace{u^\top W^k_{e} k_j}_{\text{sesgo global de contenido}} + \underbrace{v^\top W^k_{r} R_{i-j}}_{\text{sesgo global de posición}}$$

**Codificación relativa de segmento** (§2.5, aporte propio de XLNet). BERT usa embeddings absolutos $E_A$ y $E_B$. XLNet solo codifica **"mismo segmento o no"**, con dos consecuencias: mejor generalización por el sesgo inductivo relativo, y la posibilidad de fine-tunear en tareas con **más de dos segmentos** —QA multi-hop, por ejemplo— algo que BERT no puede porque solo tiene dos embeddings.

## Las tres cabezas raras de `XLNetForQuestionAnswering`

Cargar el modelo de QA produce un aviso de HuggingFace con **seis submódulos** inicializados al azar, contra los **dos vectores** de BERT. Es la diferencia arquitectónica que más sorprende al usarlo, y tiene una razón.

**`start_logits`** — `Linear(768, 1)` por token, softmax sobre el párrafo. Funcionalmente idéntica a BERT.

**`end_logits`** — aquí está la innovación:

```
end_logits.dense_0:   Linear(2 * hidden_size, hidden_size)   # [h_end, h_start] concatenados
end_logits.LayerNorm: LayerNorm(hidden_size)
end_logits.dense_1:   Linear(hidden_size, 1)
```

El final **se condiciona en la representación del token de inicio**. BERT predice ambos extremos de forma independiente, con el mismo defecto de fondo que su objetivo de enmascarado. Es un diseño estilo **R-Net**, no estilo BERT.

**`answer_class`** — una cabeza dedicada a decidir si la pregunta **tiene respuesta**, para SQuAD 2.0. Ver [SQuAD 2.0](/papers/squad2-rajpurkar-2018).

Que esos pesos aparezcan sin inicializar **es lo esperado**: el checkpoint pre-entrenado no incluye cabezas de tarea.

## Resultados y el matiz que los acompaña

XLNet supera a BERT en las 20 tareas evaluadas, con márgenes grandes en las que dependen de contexto largo: RACE, SQuAD, clasificación de documentos.

Pero el paper también entrena con **más datos** que BERT (BooksCorpus + Wikipedia + Giga5 + ClueWeb + Common Crawl, ~32,89 B subwords contra ~13 GB). Las ablaciones (§3.4) intentan aislar el efecto del objetivo controlando los datos, y el aporte de PLM sobrevive — pero es sensiblemente menor que la diferencia bruta de las tablas principales.

Ese es justamente el punto que [RoBERTa](/papers/roberta-liu-2019) explotaría meses después: buena parte de la brecha atribuida al objetivo era, en realidad, entrenamiento insuficiente de BERT.

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) carga XLNet en tres variantes y ahí aparecen dos tropiezos que conviene anticipar:

**El tokenizador necesita SentencePiece.** `XLNetTokenizer` depende del paquete [`sentencepiece`](/papers/sentencepiece-kudo-2018); sin él falla con un error críptico (`Couldn't instantiate the backend tokenizer`).

**Los tokens especiales van al final:**

```
Input:  "Hello world"
BERT:   [CLS] Hello world [SEP]        <- CLS al inicio
XLNet:  ▁Hello ▁world <sep> <cls>      <- CLS al FINAL
```

Quien escriba `last_hidden_state[:, 0, :]` esperando el vector de clasificación —el reflejo aprendido con BERT— obtendrá el primer token del texto. Hay que usar `[:, -1, :]`.

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [RoBERTa](/papers/roberta-liu-2019) · [SentencePiece](/papers/sentencepiece-kudo-2018) · [SQuAD 2.0](/papers/squad2-rajpurkar-2018) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
