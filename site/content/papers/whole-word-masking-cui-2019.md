---
title: "Pre-Training with Whole Word Masking (2019)"
weight: 299
math: true
---

{{< paper-card
    title="Pre-Training with Whole Word Masking for Chinese BERT"
    authors="Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang (Joint Laboratory of HIT and iFLYTEK Research)"
    year="2019"
    venue="IEEE/ACM TASLP 2021 / arXiv:1906.08101"
    arxiv="1906.08101"
    pdf="/papers/whole-word-masking-cui-2019.pdf" >}}
Un cambio de una línea en el procedimiento de enmascarado de [BERT](/papers/bert-devlin-2018), con efecto medible: en vez de seleccionar subwords de forma independiente, se selecciona **la palabra completa** y se enmascaran todos sus fragmentos a la vez. Es la técnica que [BETO](/papers/beto-canete-2020) adopta —el `wwm` de `bert-base-spanish-wwm-cased`— y que Google terminó incorporando a sus propios checkpoints de BERT.
{{< /paper-card >}}

---

## El problema

BERT enmascara subwords de forma independiente, cada uno con probabilidad 0,15:

$$\text{seleccionar}(t_i) \overset{\text{iid}}{\sim} \text{Bernoulli}(0{,}15), \qquad \forall i \in \{1, \ldots, k\}$$

Para una palabra que se fragmenta en $k$ subwords, la cantidad enmascarada sigue una $\text{Binomial}(k, 0{,}15)$ — casi siempre uno o ninguno, rara vez todos.

Y ahí aparece el atajo. Con `playing` → `play ##ing`, si solo se enmascara `##ing`:

```
BERT original:  the play [M]   child ##ren laughed [M] ##ly
```

Predecir `##ing` teniendo `play` visible es casi trivial: en inglés, la mayoría de las continuaciones posibles tras `play` son `##ing`, `##ed`, `##er`, `##s`. **El modelo aprende morfología de sufijos, no semántica.** Está gastando una fracción importante de su presupuesto de entrenamiento en una tarea que no requiere entender la oración.

## La corrección

Se mueve la selección al nivel de palabra y se propaga a sus fragmentos:

$$\text{seleccionar}(w) \sim \text{Bernoulli}(0{,}15), \qquad \text{seleccionar}(t_i) = \text{seleccionar}(w)\ \ \forall i$$

Todos los subwords de una palabra se enmascaran juntos, o ninguno:

```
WWM:  the [M] [M]  child ##ren laughed [M] [M]
```

Ahora predecir `play ##ing` exige inferir que el sujeto es `children`, que el tiempo verbal es progresivo, y elegir un verbo compatible con `children laughed`. **La tarea pasa de morfológica a semántica.**

{{< concept-alert type="clave" >}}
Lo notable es lo barato del cambio: no toca la arquitectura, ni la función de pérdida, ni el vocabulario, ni el conteo de parámetros. Solo cambia **cuál de los 15 % de tokens se elige**, agrupando la decisión por palabra.

Es un buen ejemplo de una clase de mejora que aparece varias veces en el diplomado: el modelo no era peor, **la tarea de entrenamiento tenía una fuga** que permitía resolverla sin aprender lo que se quería enseñar.
{{< /concept-alert >}}

## El caso chino, que es el del paper

En chino el problema es más agudo, porque el tokenizador de BERT trabaja **carácter por carácter** y una palabra suele ser de dos caracteres:

```
Oración:      使用语言模型来预测下一个词的概率
Palabras:     语言 | 模型 | 来 | 预测 | ...      (idioma | modelo | para | predecir)
Tokens BERT:  语 言 模 型 来 预 测 ...            (carácter a carácter)

BERT original:  语 言 [M] 型 来 [M] 测 ...        (enmascara medio 模型 y medio 预测)
```

Enmascarar 模 dejando 型 visible es pedirle al modelo que complete la otra mitad de una palabra bisilábica — que es, de nuevo, una tarea de forma y no de significado.

WWM en chino requiere un paso extra que en inglés no hace falta: **segmentar palabras** (los autores usan LTP), porque el chino no separa por espacios. Es importante entender que esa segmentación **solo se usa para decidir el enmascarado**: el input del modelo sigue siendo la secuencia de caracteres, así que no hay cambio en la interfaz ni riesgo de propagar errores del segmentador a inferencia.

## Adopción

La técnica se difundió más allá del chino:

- **Google** publicó `bert-large-uncased-whole-word-masking` y `bert-large-cased-whole-word-masking`, adoptando el método en sus propios checkpoints en inglés.
- **[BETO](/papers/beto-canete-2020)** lo usa — el `wwm` en `dccuchile/bert-base-spanish-wwm-cased` es exactamente esto.
- Los autores continuaron la línea con **MacBERT**, que reemplaza `[MASK]` por sinónimos reales durante el entrenamiento, atacando el otro desajuste conocido de BERT: que el token `[MASK]` aparece en pre-entrenamiento y nunca en fine-tuning.

## Limitaciones reconocibles

- **La ganancia es modesta y no uniforme.** Del orden de uno a dos puntos según la tarea, con las mejores en comprensión lectora y las más pequeñas en clasificación de oraciones.
- **Depende de la calidad de la segmentación** en idiomas sin espacios. Un segmentador mediocre agrupa mal y el beneficio se diluye.
- **Se confunde con span masking.** WWM agrupa por **palabra**; SpanBERT enmascara **secuencias contiguas de longitud aleatoria**, que pueden cruzar fronteras de palabra. Son técnicas distintas con motivaciones distintas.
- **No corrige el desajuste del token `[MASK]`**, que es lo que motivó tanto a MacBERT como, por otra vía, a [XLNet](/papers/xlnet-yang-2019).

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) usa BETO, cuyo nombre de checkpoint contiene `wwm`. Es una de esas piezas de nomenclatura que se copian sin leer: identificar qué significa ese fragmento —y que remite a este paper— es parte de entender qué modelo se está cargando.

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [BETO](/papers/beto-canete-2020) · [RoBERTa](/papers/roberta-liu-2019) · [XLNet](/papers/xlnet-yang-2019) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
