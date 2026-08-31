---
title: "SentencePiece: Language-Independent Subword Tokenizer (2018)"
weight: 300
math: true
---

{{< paper-card
    title="SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
    authors="Taku Kudo, John Richardson (Google)"
    year="2018"
    venue="EMNLP 2018 (System Demonstrations) / arXiv:1808.06226"
    arxiv="1808.06226"
    pdf="/papers/sentencepiece-kudo-2018.pdf" >}}
Un *system demonstration paper* de ocho páginas que terminó siendo infraestructura invisible de casi todo el NLP moderno. Su aporte no es un algoritmo sino una **decisión de representación**: tratar el texto como una secuencia de codepoints Unicode donde **el espacio es un símbolo más**. Eso elimina la pre-tokenización por palabras, hace la detokenización reversible sin heurísticas, y funciona igual en inglés, chino o japonés. Es la dependencia que el [Laboratorio 20](/laboratorios/lab-20) descubre a la mala cuando `XLNetTokenizer` falla al cargar.
{{< /paper-card >}}

---

## El problema que resuelve

Antes de 2018, tokenizar era una cadena frágil de herramientas específicas por idioma: Moses para inglés, MeCab o KyTea para japonés, jieba para chino, y después un tokenizador de subwords como `subword-nmt`. Cada eslabón con sus reglas, sus casos límite y su comportamiento propio.

Y había un problema más profundo: **la detokenización no era reversible**. Recuperar el texto original desde los tokens exigía reglas heurísticas distintas por idioma —dónde va el espacio antes de una coma, cómo se recompone una contracción— y esas reglas fallan.

## La decisión de representación

SentencePiece trata la entrada como un flujo de codepoints Unicode, **incluido el espacio**, que se escapa con el meta-símbolo `▁` (U+2581):

```
Texto crudo:  Hello world.
Interno:      ▁Hello▁world.
Tokenizado:   [▁Hello] [▁world] [.]
```

De ahí sale la propiedad que lo hizo universal: la detokenización es una línea, sin reglas y sin idioma.

```python
detok = ''.join(tokens).replace('▁', ' ')
```

Compárese con la convención de `subword-nmt`, que marca continuaciones intra-palabra con `@@`:

```
subword-nmt:     [Hello] [wor@@] [ld] [.]
SentencePiece:   [▁Hello] [▁wor] [ld] [.]
```

La diferencia práctica es mayor de lo que parece: `subword-nmt` **no puede representar espacios consecutivos**, porque asume un espacio implícito entre tokens completos. SentencePiece sí, porque cada espacio es un `▁` explícito. Importa en cuanto el texto sea código fuente, tablas o poesía con espaciado significativo.

{{< concept-alert type="clave" >}}
**Sin pre-tokenización por palabras.** Como opera sobre Unicode crudo, no necesita un segmentador previo. Eso es lo que lo vuelve genuinamente independiente del idioma: para chino o japonés —que no separan por espacios— desaparece el eslabón más frágil de la cadena, y con él la posibilidad de que un error de segmentación se propague a todo lo que venga después.
{{< /concept-alert >}}

## BPE y Unigram LM

SentencePiece es una **librería** que implementa dos algoritmos de construcción de vocabulario, y conviene no confundir la herramienta con el algoritmo:

| Algoritmo | Criterio | Marcador | Quién lo usa |
|---|---|---|---|
| **WordPiece** | Maximizar verosimilitud de un LM unigrama | `##` continuación | [BERT](/papers/bert-devlin-2018), [BETO](/papers/beto-canete-2020) |
| **BPE** | Fusionar el par más frecuente | `Ġ` o `▁` inicio | [GPT-2](/papers/gpt-2-radford-2019), [RoBERTa](/papers/roberta-liu-2019) |
| **Unigram LM** | Podar desde un vocabulario grande maximizando verosimilitud | `▁` inicio | [XLNet](/papers/xlnet-yang-2019), T5, ALBERT |

El **Unigram LM** —introducido por Kudo en un paper anterior de 2018— trabaja al revés que BPE: parte de un vocabulario grande y **poda** los símbolos cuya eliminación menos degrada la verosimilitud del corpus. La diferencia práctica es que asigna una probabilidad a cada segmentación posible, en lugar de producir una única segmentación determinista.

Eso habilita la **regularización por subwords**: muestrear segmentaciones distintas de la misma palabra durante el entrenamiento, como aumentación de datos sobre la tokenización misma.

## Una confusión frecuente

Decir "el modelo usa SentencePiece" no dice qué algoritmo usa: puede ser BPE o Unigram. Y a la inversa, un vocabulario construido con SentencePiece puede exportarse a otro formato — es exactamente lo que ocurrió con [BETO](/papers/beto-canete-2020), cuyo vocabulario se entrenó con SentencePiece BPE y se publicó en formato WordPiece con marcadores `##` para que fuese compatible *drop-in* con `BertTokenizer`.

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) tropieza con esta librería antes de entender qué es. Cargar `XLNetTokenizer` sin tener `sentencepiece` instalado falla con un mensaje que no ayuda:

```
Couldn't instantiate the backend tokenizer
```

La razón es que HuggingFace **no** incluye `sentencepiece` entre sus dependencias obligatorias: es un paquete con extensión en C++ que agrega peso a la instalación, y solo lo necesitan los tokenizadores que se construyeron con él. Los modelos que usan WordPiece o BBPE funcionan sin ella.

El arreglo es `pip install sentencepiece`, y el aprendizaje es que **la elección de tokenizador arrastra dependencias de sistema**, no solo diferencias de vocabulario.

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [XLNet](/papers/xlnet-yang-2019) · [RoBERTa](/papers/roberta-liu-2019) · [BETO](/papers/beto-canete-2020) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
