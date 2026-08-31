---
title: "El tokenizador cruzado"
weight: 2
math: true
---

La pregunta 9 del laboratorio plantea un experimento mental:

> *"Cada modelo trae su propio tokenizador. ¿Qué efecto tendría si usáramos el tokenizador de un modelo (ejemplo: el de BERT) para otro (ejemplo: para GPT2)? (…) asumiendo que solo se usará en modo inferencia (…) **y que el código ejecuta sin errores**."*

Esa última cláusula es la parte interesante del enunciado, y conviene detenerse en ella: la pregunta **da por supuesto que no hay excepción**. Está preguntando por un modo de falla **silencioso**.

Durante la ejecución del laboratorio el experimento ocurre solo, por accidente: queda un `XLNetTokenizer` en memoria de las celdas 12-14 y se usa para decodificar salidas de RoBERTa. El resultado es texto basura, sin una sola traza de error.

## Por qué no explota

Un modelo de lenguaje no recibe texto: recibe **enteros**. La capa de embeddings es una tabla de búsqueda `nn.Embedding(vocab_size, hidden_size)`, y lo único que puede fallar es que un índice quede fuera de rango.

$$\text{tokenizador} : \text{texto} \longrightarrow \mathbb{Z}^n \qquad\qquad \text{modelo} : \mathbb{Z}^n \longrightarrow \mathbb{R}^{n \times H}$$

**Los dos extremos no se comunican.** El modelo no tiene forma de saber que el entero 2.847 significaba `##ing` en el vocabulario que lo produjo y significa `▁cathedral` en el suyo. Busca la fila 2.847 de su tabla y sigue.

Se obtiene la matriz de embeddings correcta, las capas de atención correctas y salidas de la forma correcta — que representan un texto que nadie escribió.

{{< concept-alert type="clave" >}}
La condición que decide si hay excepción o no es puramente aritmética:

$$\max(\text{ids del tokenizador A}) < \text{vocab\_size del modelo B} \;\Rightarrow\; \textbf{falla silenciosa}$$

$$\max(\text{ids del tokenizador A}) \geq \text{vocab\_size del modelo B} \;\Rightarrow\; \texttt{IndexError}$$

Con los tamaños de los modelos del laboratorio:

| Tokenizador | Vocabulario | → modelo con vocab 30.522 (BERT) | → modelo con vocab 50.265 (RoBERTa) |
|---|---:|---|---|
| BERT | 30.522 | — | silencioso |
| BETO | 31.002 | `IndexError` probable | silencioso |
| XLNet | 32.000 | `IndexError` probable | silencioso |
| RoBERTa | 50.265 | `IndexError` casi seguro | — |

**Cruzar hacia un modelo de vocabulario más grande casi siempre falla en silencio. Cruzar hacia uno más chico tiende a explotar.** Y la diferencia entre las dos situaciones no dice nada sobre cuál de los dos errores es más grave: son el mismo error, y el que avisa es el afortunado.
{{< /concept-alert >}}

## Qué se rompe exactamente

Tres cosas a la vez, y ninguna produce señal:

**1 · Los tokens no significan lo mismo.** El identificador 2.847 apunta a subwords distintos en cada vocabulario. Toda la secuencia queda traducida a un texto arbitrario.

**2 · Los tokens especiales quedan en el lugar equivocado.** Ya vimos los valores en [las trece preguntas](03-las-trece-preguntas): `[CLS]` es 101 en BERT, `<s>` es 0 en RoBERTa y `[CLS]` es 4 en BETO. Un `[CLS]` de BERT insertado en una secuencia para RoBERTa es, para RoBERTa, el token 101 — un subword cualquiera, no una marca de inicio.

**3 · Las convenciones de segmentación difieren.** WordPiece marca continuación con `##`, BBPE marca inicio de palabra con `Ġ`, SentencePiece con `▁`. Ni siquiera la noción de "dónde empieza una palabra" coincide.

## Cómo se ve

Decodificar salidas de un modelo con el tokenizador de otro produce texto que **parece** texto: tiene espacios, tiene fragmentos de palabras reales, tiene puntuación. No parece un error, parece un modelo funcionando mal.

Y ahí está el riesgo real: en un pipeline con métricas automáticas, un cruce de tokenizadores baja el rendimiento sin dejar traza. Se busca el problema en los hiperparámetros, en los datos, en la arquitectura — en todas partes menos en la línea que cargó el tokenizador equivocado.

## La práctica defensiva

```python
# frágil: dos strings que hay que mantener sincronizados a mano
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model     = RobertaModel.from_pretrained('roberta-base')

# robusto: un solo string, imposible desincronizar
CKPT = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model     = AutoModel.from_pretrained(CKPT)
```

Las clases `Auto*` no son azúcar sintáctica: leen el `config.json` del checkpoint y **eligen la clase correcta**. Con un único identificador de checkpoint compartido, el error de esta página es estructuralmente imposible.

Un chequeo barato para cuando la carga no puede unificarse:

```python
assert model.config.vocab_size == tokenizer.vocab_size, \
    f"desajuste: modelo {model.config.vocab_size} vs tokenizador {tokenizer.vocab_size}"
```

No detecta todos los cruces —dos vocabularios pueden tener el mismo tamaño y contenido distinto— pero atrapa la mayoría de los casos reales por un costo de una línea.

## Por qué esto importa más allá del laboratorio

El notebook trabaja con cinco modelos en un mismo espacio de nombres y reasigna la variable `tokenizer` seis veces. Es un entorno especialmente propicio para el error — pero la lección no es sobre notebooks.

Es un caso de una familia más amplia: **dos componentes acoplados por una convención que ninguno de los dos verifica**. Aparece cada vez que un identificador numérico cruza una frontera de sistema sin llevar consigo el esquema que lo interpreta.

---

**Siguiente:** [Las trece preguntas](03-las-trece-preguntas) — las respuestas del laboratorio y qué hay que mirar en cada una.
