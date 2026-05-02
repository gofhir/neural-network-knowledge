---
title: "BPE (Byte Pair Encoding)"
weight: 290
math: true
---

**Byte Pair Encoding (BPE)** es el algoritmo de **subword tokenization** que sostiene la era moderna de LLMs: GPT-2/3/4, Llama, Mistral, Falcon y la mayoria de los modelos abiertos lo usan (en variantes) como puerta de entrada al modelo. Fue adaptado a NLP por **Sennrich, Haddow y Birch (ACL 2016)** desde el algoritmo de compresion homonimo de **Gage (1994)**, y resuelve un dilema viejo: como representar texto numericamente sin caer ni en char-level (sin semantica, secuencias largas) ni en word-level (out-of-vocabulary masivo, vocabularios gigantes).

Es facil de subestimar: parece "solo" un preprocesador. Pero la eleccion del tokenizador determina el tamano de la embedding table, la longitud efectiva de las secuencias, el costo de inferencia por idioma, y el sesgo lingüistico del modelo. En el [Camino 2.5](/clases/clase-14/practica/30-bpe-desde-cero/) lo implementamos desde cero precisamente porque sin entenderlo no se entiende por que GPT-4 cobra el doble por tokenizar portugues que ingles.

---

## 1. El problema de vocabulario

Una red neuronal no opera sobre strings: opera sobre vectores. La pregunta es **como discretizar texto** a un alfabeto finito $V = \{1, 2, \ldots, |V|\}$ sobre el cual definir embeddings $E \in \mathbb{R}^{|V| \times d}$. Tres caminos:

| Esquema | Vocab tipico | OOV | Longitud secuencia | Semantica por token |
|---------|------------|-----|-------------------|---------------------|
| Char-level | 50-200 | Cero | $\sim 5\times$ palabras | Casi nula |
| Word-level | 50k-500k | Catastrofico | 1$\times$ palabras | Alta |
| Subword (BPE) | 1k-100k | Cero (con bytes) | 1.3-2$\times$ palabras | Intermedia |

**Char-level** ("Shakespearean" $\to$ ['S','h','a','k','e',...]) tiene vocab pequeno y nunca falla, pero le pide al modelo que aprenda morfologia desde cero y multiplica por 5 la cantidad de pasos de atencion. **Word-level** colapsa cuando aparece "Shakespearean" sin haber estado en el corpus de entrenamiento (queda como `<UNK>`). El **subword** parte la palabra en piezas reutilizables: "Shake" + "spear" + "ean", y cada pieza ya estaba en el vocab porque viene de palabras frecuentes ("Shake" de "Shakespeare", "ean" de "European", "Korean", etc.).

Ese es el sweet spot que BPE captura.

---

## 2. La idea de BPE

El nucleo es **greedy y data-driven**: empezar con un vocabulario de caracteres atomicos y, observando el corpus, **fundir iterativamente el par de tokens consecutivos mas frecuente** en un nuevo simbolo. Repetir hasta alcanzar un vocab target $|V|$.

**Ejemplo de juguete** sobre el corpus `"aaabdaaabac"` con vocab inicial $\{a, b, c, d\}$:

- Iter 1: pares y frecuencias: $(a,a)=4$, $(a,b)=2$, $(b,d)=1$, $(d,a)=1$, $(b,a)=1$, $(a,c)=1$. Ganador: $(a,a)$. Merge $\to$ `aa`. Corpus: `aa ab d aa ab ac` (separadores virtuales).
- Iter 2: $(aa, b)=2$ es el mas frecuente. Merge $\to$ `aab`. Corpus: `aab d aab ac`.
- Iter 3: $(aab, d)=1$, $(d, aab)=1$, $(aab, ac)=1$ — empate, elegir cualquiera; supongamos `aab d`. Corpus: `aabd aab ac`.

Despues de 3 merges el vocabulario es $\{a, b, c, d, aa, aab, aabd\}$. Una palabra como `"aabd"` que originalmente requeria 4 tokens ahora es **1**.

En texto natural el efecto es dramatico: tras unos pocos miles de merges los morfemas comunes (`-ing`, `-ed`, `-tion`, `un-`) se vuelven tokens unicos, y palabras enteras frecuentes (`the`, `and`, `because`) tambien.

---

## 3. El algoritmo formal

**Entrenamiento**:

```python
def bpe_train(corpus: str, num_merges: int):
    vocab = set(corpus)                       # caracteres unicos
    tokens = list(corpus)                     # secuencia inicial: chars
    merges = []
    for _ in range(num_merges):
        pair_counts = count_consecutive_pairs(tokens)
        if not pair_counts: break
        best_pair = argmax(pair_counts)
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
        merges.append(best_pair)
        tokens = apply_merge(tokens, best_pair, new_token)
    return vocab, merges
```

**Encoding** de un texto nuevo:

```python
def encode(text: str, vocab, merges):
    tokens = list(text)
    for merge in merges:                       # IMPORTANTE: en orden
        tokens = apply_merge(tokens, merge[0], merge[1])
    return [vocab[t] for t in tokens]
```

El detalle critico es que **los merges deben aplicarse en el mismo orden en que se aprendieron**. Los merges tempranos crean los tokens que merges posteriores combinan: si BPE aprendio primero `t+h $\to$ th` y despues `th+e $\to$ the`, aplicar el segundo antes del primero romperia la cadena. Es la diferencia entre un tokenizador determinista y uno con bugs sutiles.

Complejidad: O(num_merges $\cdot$ |corpus|) en la version naive; existen optimizaciones con priority queues e indices invertidos que la bajan a casi lineal.

---

## 4. Por que vocab size importa para el modelo

La embedding table tiene exactamente $|V| \cdot d$ parametros, replicada en la capa de salida (lm_head). Es decir:

$$\text{params}_\text{embed} = 2 \cdot |V| \cdot d_\text{model}$$

(o $|V| \cdot d$ si se hace **weight tying** entre embedding e lm_head, practica estandar).

Para un modelo Transformer chico de $d_\text{model}=128$:

| Vocab | Params embedding | Params modelo total | % embedding |
|-------|------------------|---------------------|-------------|
| 65 (char) | 8.3 k | 1.1 M | 0.8 % |
| 1112 (BPE pequeno) | 142 k | 1.2 M | 12 % |
| 50 257 (GPT-2) | 6.4 M | $\sim$ 124 M | 5 % |

Regla de pulgar: **el embedding no debe dominar al modelo**. Si la embedding table excede el 30% de los parametros, el modelo gasta capacidad en memorizar identidades de tokens raros en vez de aprender estructura. Para nuestros minis (1-5 M params) un vocab BPE de 1k-2k es lo apropiado; para modelos de 1B+, vocabs de 32k-130k son comunes.

---

## 5. BPE en produccion: GPT, Llama, Mistral

Los grandes laboratorios usan BPE con variantes propietarias:

| Modelo | Tokenizador | Vocab | Notas |
|--------|-------------|-------|-------|
| GPT-2/3 | tiktoken (BPE byte-level) | 50 257 | Opera sobre bytes, no Unicode |
| GPT-4 | cl100k_base | 100 256 | Mejor compresion en codigo y multi-idioma |
| Llama 1/2 | SentencePiece + BPE | 32 000 | Normalizacion Unicode previa |
| Llama 3 | tiktoken-style | 128 256 | Salto a vocab grande para multilingüe |
| Mistral | Tekken | 130 000 | BPE optimizado para codigo + 100 idiomas |
| BERT | WordPiece | 30 522 | Variante de BPE, no exactamente BPE |

**Byte-level BPE** (GPT-2 en adelante) es una variante clave: en vez de operar sobre caracteres Unicode, opera sobre los **256 bytes** crudos. Esto garantiza que **cualquier** entrada — emojis, glifos raros, bytes corruptos — sea tokenizable sin `<UNK>`. El precio es que un caracter chino (3 bytes en UTF-8) puede ocupar 3 tokens iniciales antes de que los merges los fusionen.

---

## 6. BPE vs WordPiece vs SentencePiece

Tres familias relacionadas pero distintas:

- **BPE original (Sennrich)**: scoring por **frecuencia** del par. Usado por GPT-2/3/4, RoBERTa.
- **WordPiece** (Schuster & Nakajima 2012, popularizado por BERT): scoring por **likelihood ratio** $\log P(ab) - \log P(a) - \log P(b)$, no frecuencia bruta. Genera prefijo `##` para subwords no-iniciales (`playing` $\to$ `play` + `##ing`). Usado por BERT, DistilBERT, Electra.
- **SentencePiece** (Kudo & Richardson 2018): no es un algoritmo sino un **framework**. Soporta tanto BPE como **Unigram LM** (Kudo 2018) bajo la misma interfaz. Su valor real es operar **directamente sobre raw text** sin pre-tokenizacion en palabras — crucial para idiomas sin espacios (japones, chino, tailandes). Usado por T5, mT5, Llama 1/2, ALBERT, XLNet.

Trade-off practico: BPE/WordPiece asumen separadores de palabra explicitos y fallan en idiomas que no los tienen. SentencePiece + Unigram LM es mas robusto cross-lingual pero ligeramente mas costoso de entrenar.

---

## 7. Implementacion PyTorch (snippet del Camino 2.5)

Version simplificada del [BPE desde cero](/clases/clase-14/practica/30-bpe-desde-cero/):

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    def train(self, corpus: str, num_merges: int) -> None:
        for c in sorted(set(corpus)):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
                self.id_to_token[self.vocab[c]] = c

        tokens = list(corpus[:50_000])
        for _ in range(num_merges):
            counts = Counter()
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += 1
            if not counts:
                break

            (a, b), _ = counts.most_common(1)[0]
            new_token = a + b
            self.merges.append((a, b))
            self.vocab[new_token] = len(self.vocab)
            self.id_to_token[self.vocab[new_token]] = new_token
            tokens = self._apply_merge(tokens, a, b, new_token)

    def encode(self, text: str) -> list[int]:
        tokens = [c for c in text if c in self.vocab]
        for a, b in self.merges:
            new_token = a + b
            if new_token in self.vocab:
                tokens = self._apply_merge(tokens, a, b, new_token)
        return [self.vocab[t] for t in tokens if t in self.vocab]
```

El metodo `_apply_merge` recorre la lista y reemplaza ocurrencias consecutivas `(a, b)` por `new_token`. La version completa del repo agrega bos/eos tokens, manejo de bytes, y serializacion JSON.

---

## 8. Limitaciones de BPE

Pese a su exito, BPE arrastra patologias bien documentadas:

- **Tokenizacion suboptima sin separadores**: japones, chino y tailandes sufren porque BPE asume que las palabras estan delimitadas por espacios. SentencePiece byte-level mitiga esto pero no lo elimina.
- **Sesgo del corpus**: un BPE entrenado sobre 95% ingles tokenizara espanol con peor compresion. En el Camino 2.5 medimos esto: con BPE entrenado sobre Quijote (espanol), el ratio chars-por-token cae a $\sim$ 0.58, mientras que un texto ingles con el mismo tokenizador rinde apenas $\sim$ 0.38. Cambia drasticamente el costo computacional.
- **Aritmetica rota**: la cadena `"12345"` puede fragmentarse como `"12"` + `"345"` o `"1234"` + `"5"` segun los merges aprendidos. Esto **destruye la estructura posicional** de los digitos y es la causa raiz de que GPT-3.5 fallara en sumas largas. Modelos modernos (Llama 3, GPT-4o) fuerzan **digit-level tokenization** para numeros como mitigacion.
- **Fertility cross-lingual**: idiomas morfologicamente ricos (turco, finlandes, hungaro) generan secuencias mucho mas largas. La OpenAI cobra por token, asi que esto se traduce en **discriminacion economica** de hablantes no-ingleses, ademas de degradacion de calidad por mayor consumo de context window.

{{< concept-alert type="clave" >}}
**BPE no es neutral.** El corpus de entrenamiento del tokenizador determina que tan bien tokenizara cada idioma y dominio. GPT-4 tokeniza chino con $\sim$ 2-3 chars/token y portugues con $\sim$ 5 chars/token — el ingles esta privilegiado por construccion. Esto se traduce en **costos de API mas altos** para usuarios no-ingleses, **menos contenido por context window**, y **calidad de modelado degradada** para idiomas subrepresentados. La eleccion del corpus es una decision politica, no solo tecnica.
{{< /concept-alert >}}

---

## 9. Resumen

- **BPE** es un algoritmo greedy de subword tokenization que aprende un vocabulario fundiendo iterativamente los pares de tokens mas frecuentes del corpus.
- Resuelve el dilema char-level (sin semantica, secuencias largas) vs word-level (OOV catastrofico, vocab gigante) con un punto medio: vocab de 1k-130k segun escala del modelo.
- Es **el** algoritmo de tokenizacion que permitio escalar Transformers a la era de LLMs: GPT-2/3/4, Llama, Mistral, RoBERTa todos lo usan en variantes.
- WordPiece (BERT) y SentencePiece (T5, Llama) son primos cercanos con diferencias en el scoring y el manejo de pre-tokenizacion.
- Limitaciones reales: sesgo del corpus, idiomas sin espacios, aritmetica fragmentada, fertility cross-lingual asimetrica. La eleccion del corpus de entrenamiento del tokenizador es una decision con consecuencias visibles para usuarios reales.

Sin BPE no hay LLMs a la escala actual. Pero con BPE heredamos sus sesgos: cualquier proyecto serio de modelo multilingüe debe **auditar el tokenizador** antes que la arquitectura.

---

### Ver tambien

- [SFT](/fundamentos/sft) — el siguiente paso despues de elegir tokenizador
- [DPO](/fundamentos/dpo) — preferencias sobre modelos pretrained
- [Self-attention](/fundamentos/self-attention) — el bloque que opera sobre los tokens producidos por BPE
- [Transformer](/fundamentos/transformer) — la arquitectura que usa BPE en produccion
- [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) — que aprende la embedding table que indexamos con BPE
- [Camino 2.5 cap 30 — BPE desde cero](/clases/clase-14/practica/30-bpe-desde-cero/) — implementacion paso a paso
- [Camino 2.5 cap 31 — Pretrain con BPE](/clases/clase-14/practica/31-pretrain-bpe/) — efecto en el modelo
