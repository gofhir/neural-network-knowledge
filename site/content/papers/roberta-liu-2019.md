---
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)"
weight: 297
math: true
---

{{< paper-card
    title="RoBERTa: A Robustly Optimized BERT Pretraining Approach"
    authors="Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov (Facebook AI / University of Washington)"
    year="2019"
    venue="arXiv:1907.11692"
    arxiv="1907.11692"
    pdf="/papers/roberta-liu-2019.pdf" >}}
No propone ninguna arquitectura nueva. Toma [BERT](/papers/bert-devlin-2018) exactamente como está, cambia **cómo se entrena**, y supera a [XLNet](/papers/xlnet-yang-2019) — que sí traía un objetivo nuevo. Su conclusión es incómoda y quedó como una de las lecciones duraderas del campo: **BERT estaba significativamente subentrenado**, y buena parte de las mejoras que en 2019 se atribuían a la arquitectura eran, en realidad, mejor régimen de entrenamiento. Aparece en el [Laboratorio 20](/laboratorios/lab-20).
{{< /paper-card >}}

---

## El resultado que reordena la narrativa

La lectura de la comunidad a mediados de 2019 era que XLNet le ganaba a BERT porque el *permutation language modeling* era mejor objetivo que el enmascarado. El paper de XLNet reportaba 89,8 en MNLI contra 86,6 de BERT: una brecha aparentemente arquitectónica.

La tabla 5 de RoBERTa desmonta esa lectura controlando por corpus:

| Modelo | Datos | MNLI (Acc) |
|---|---|---:|
| BERT-large | 13 GB | 86,6 |
| XLNet-large | 13 GB | 88,4 |
| **RoBERTa** (mismo corpus que BERT) | 13 GB | **89,0** |
| XLNet-large | 126 GB | 89,8 |
| **RoBERTa** (corpus completo) | 160 GB | **90,2** |

{{< concept-alert type="clave" >}}
**Restringido al corpus original de BERT —los mismos 13 GB—, RoBERTa supera a XLNet entrenado sobre ese mismo corpus por 0,6 puntos.** La ganancia atribuida al *permutation LM* se evapora al controlar por régimen de entrenamiento. Lo que queda de mejora viene de más datos, y eso es ortogonal a la arquitectura.

Es el resultado que convierte a RoBERTa en el paper definitorio de la era encoder-only tardía: en 2019 **la frontera ya no era arquitectónica, sino de datos y cómputo**.
{{< /concept-alert >}}

## Las cinco modificaciones

**1 · Eliminar Next Sentence Prediction.** BERT optimiza $\mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$. RoBERTa quita el segundo término y **mejora** en tareas donde NSP supuestamente ayudaba. La ablación es explícita: entrenar con oraciones contiguas de un mismo documento, sin la tarea de clasificación de pares, rinde igual o mejor.

**2 · Enmascarado dinámico.** BERT genera las máscaras **una vez**, al preprocesar. Para entrenar 40 épocas duplica el dataset 10 veces con máscaras distintas, así que cada patrón se ve 4 veces. RoBERTa genera la máscara **al vuelo en cada forward**: sobre 500 K pasos con batch de 8 K, cada token enmascarable ve miles de millones de patrones distintos en lugar de diez.

**3 · Más datos: 16 GB → 160 GB.** BooksCorpus y Wikipedia, más CC-News, OpenWebText y Stories.

**4 · Batches masivos: 256 → 8 K secuencias.** De 131 K tokens por batch a más de 4 M, en la línea del trabajo contemporáneo sobre entrenamiento con batches grandes.

**5 · Más pasos efectivos.** BERT: 1 M pasos × batch 256 ≈ **131 B tokens vistos**. RoBERTa: 500 K pasos × batch 8 K ≈ **2 T tokens** — unas **16 veces más**, con la mitad de los pasos nominales.

## El cambio de tokenizador

RoBERTa reemplaza el WordPiece de 30 K de BERT por **Byte-level BPE** de 50 K, heredado de [GPT-2](/papers/gpt-2-radford-2019). La propiedad decisiva: al operar sobre bytes, **el vocabulario nunca produce `<unk>`** — cualquier cadena, en cualquier idioma o alfabeto, se representa.

El costo es un vocabulario un 66 % mayor, que se paga sobre todo en la matriz de embeddings. El paper reconoce que la comparación no es completamente limpia, porque BBPE rinde ligeramente peor en algunas tareas y el cambio se adopta igual por su robustez.

## La trampa al portar código

Los tokens especiales cambian de nombre **y de identificador**, y esto muerde a quien traiga hábitos de BERT:

| Función | BERT | RoBERTa | ID en BERT-base | ID en RoBERTa-base |
|---|---|---|---:|---:|
| Clasificación / inicio | `[CLS]` | `<s>` | 101 | **0** |
| Separador / fin | `[SEP]` | `</s>` | 102 | **2** |
| Padding | `[PAD]` | `<pad>` | 0 | **1** |
| Desconocido | `[UNK]` | `<unk>` | 100 | 3 |
| Enmascarado | `[MASK]` | `<mask>` | 103 | **50264** |

Tres detalles que producen errores silenciosos:

- **El ID de `<pad>` es 1, no 0.** El 0 está tomado por `<s>`. La convención de `attention_mask` (0 = padding) se mantiene, pero rellenar tensores a mano con `torch.zeros` inserta `<s>` repetido en lugar de padding.
- **`<mask>` está al final del vocabulario** (50264), no cerca del inicio. Cualquier análisis del embedding de la máscara cambia de índice.
- RoBERTa **no usa `token_type_ids`**. Al eliminar NSP no hay segmentos A/B que distinguir, y pasarlos produce un error o se ignoran según la versión.

## Limitaciones reconocibles

- **Los factores no se aíslan del todo.** Las cinco modificaciones se evalúan parcialmente por separado, pero la configuración final las combina; cuánto aporta cada una en presencia de las demás queda sin medir.
- **El costo no se reporta con transparencia.** Entrenar RoBERTa-large exigió 1024 GPUs V100 por aproximadamente un día. La conclusión "BERT estaba subentrenado" es correcta y, a la vez, solo accionable con un presupuesto que casi nadie tiene.
- **BBPE se adopta pese a rendir algo peor** en varias tareas, por robustez. Es una decisión defendible que el paper no oculta, pero que enturbia la comparación con BERT.

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) usa `roberta-base` en la primera actividad, y ahí aparecen dos de los tropiezos anteriores: el modelo **no acepta `token_type_ids`**, y el vector de clasificación se obtiene de `<s>` (índice 0) y no de un `[CLS]` con ID 101.

El laboratorio también deja ver, por accidente, la consecuencia de mezclar tokenizador y modelo de checkpoints distintos — el tema de una de sus preguntas conceptuales, tratado en [las trece preguntas](/laboratorios/lab-20/03-las-trece-preguntas).

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [XLNet](/papers/xlnet-yang-2019) · [GPT-2](/papers/gpt-2-radford-2019) (de donde viene el BBPE) · [SentencePiece](/papers/sentencepiece-kudo-2018) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
