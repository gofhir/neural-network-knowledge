---
title: "Estadísticas de texto: FreqDist, Zipf, Heaps"
weight: 20
math: true
---

Cubre las celdas 5-20 del notebook. NLTK provee el setup para hacer análisis estadístico de corpus literarios (Moby Dick, Sense and Sensibility, Inaugural Addresses, etc.) y ver en vivo dos **leyes empíricas universales** del lenguaje natural: Zipf (frecuencias) y Heaps (crecimiento de vocabulario).

Para discusión amplia del bloque BoW que las usa, ver [Bag of Words](/fundamentos/bag-of-words).

---

## 1. `nltk.book` y los 9 textos preempaquetados

```python
from nltk.book import *
```

Esto carga 9 textos como variables globales:

| Variable | Texto | Tokens |
|---|---|---|
| `text1` | Moby Dick by Herman Melville (1851) | 260,819 |
| `text2` | Sense and Sensibility by Jane Austen (1811) | ~142k |
| `text3` | The Book of Genesis | ~44k |
| `text4` | Inaugural Address Corpus (Washington 1789 → Biden 2021) | ~150k |
| `text5` | Chat Corpus (NPS Chat) | ~45k |
| `text6` | Monty Python and the Holy Grail (script) | ~17k |
| `text7` | Wall Street Journal (Penn Treebank sample) | ~100k |
| `text8` | Personals Corpus | ~4k |
| `text9` | The Man Who Was Thursday (Chesterton 1908) | ~70k |

`from nltk.book import *` **viola buenas prácticas** Python (imports con `*`, globals) pero es deliberado por **pedagogía** — el primer día de clase un estudiante puede escribir `text1.concordance("whale")` sin entender qué es un módulo.

Para producción real, cargá explícitamente con `nltk.corpus.gutenberg.words(...)`.

---

## 2. Concordance — KWIC

```python
text1.concordance("Whales", width=80, lines=20)
```

KWIC (Key Word In Context) = listar todas las ocurrencias de una palabra con su contexto. Es la **unidad básica de la lingüística filológica desde el siglo XIII** — la primera concordance computada manualmente fue de la Vulgata bíblica (1247, 470 monjes, 10 años de trabajo).

Output esperado sobre Moby Dick:

```
Displaying 20 of 268 matches:
king up whatever random allusions to whales he could anyways find in any book wh
! EXTRACTS . " And God created great whales ." -- GENESIS . " Leviathan maketh a
st fishes that are : among which the Whales and Whirlpooles called Balaene , tak
...
```

268 matches para "Whales" en el libro → ~0.1% del texto es literalmente esa palabra. Moby Dick está **saturado de su tema central**.

**Detalle**: los primeros 20 matches están todos en la sección "EXTRACTS" (prólogo de citas literarias e históricas sobre ballenas que Melville compiló antes del capítulo 1). `concordance` recorre el texto en orden secuencial.

---

## 3. Dispersion plot — distribución temporal

```python
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
```

Para `text4` (200 años de discursos inaugurales presidenciales USA), produce un gráfico donde cada palabra recibe una línea horizontal con marcas verticales por cada ocurrencia:

```
                Lexical Dispersion Plot
  America   ||||||  || ||| |  |  || ||||| || |||| ||||||||||||||
   duties   | || |||  ||| ||  |  |       |       |
  freedom   |     |  |     |      ||| ||||| || | | || ||||||||||
democracy   |                |    | |||| ||| | ||| |||| |||| |||
 citizens   |||||| || |||||| |||  |  | ||| | | |  |  ||
            └──────────────────────────────────────────────────────
            1789                                              2021
```

**Lo que revela**:
- `citizens`: denso en siglos XVIII-XIX (republicanismo clásico), menos en XX.
- `democracy`: **casi ausente** en los primeros 100 años (Founding Fathers preferían "republic"), masivo desde Wilson (1913) y FDR (1933+).
- `freedom`: clusters en Lincoln (Guerra Civil), Wilson/FDR (Guerras Mundiales), Kennedy/Reagan (Guerra Fría), Bush (post-9/11).
- `duties`: denso en siglo XIX (lenguaje moralizante), ralea hacia el XX.

**Una sola visualización** muestra la **evolución del discurso político de USA en 200 años**. Es una de las representaciones más densas en información que produce el NLP clásico.

---

## 4. FreqDist y la Ley de Zipf

### Construir distribución de frecuencias

```python
fdist1 = FreqDist(text1)
fdist1['the']        # → 13721 ocurrencias en Moby Dick
fdist1.plot(50)      # gráfico de las top-50 palabras
```

`FreqDist` es **subclase de `collections.Counter`** con métodos extra (`N()`, `B()`, `freq()`, `hapaxes()`, `most_common()`, `plot()`).

### La Ley de Zipf

> Si ordenás las palabras por frecuencia descendente, $f(r) \propto K/r^\alpha$ con $\alpha \approx 1$.

La palabra más frecuente aparece el doble que la segunda, el triple que la tercera, etc. **Universal en cualquier idioma humano**.

Top-10 de Moby Dick (aproximado):

| Rango | Palabra | Frecuencia |
|---|---|---|
| 1 | `,` | ~18,700 |
| 2 | `the` | ~13,721 |
| 3 | `.` | ~6,800 |
| 4 | `of` | ~6,600 |
| 5 | `and` | ~6,000 |
| 6 | `a` | ~4,600 |
| 7 | `to` | ~4,500 |
| 8 | `;` | ~4,100 |
| 9 | `in` | ~3,900 |
| 10 | `that` | ~2,900 |

Predicción Zipf con $K \approx 18700$ y $\alpha = 1$: rango 2 esperado = 9350. Real: 13721. Aproximación cruda pero captura la forma.

### Por qué Zipf importa

Es la **justificación matemática de prácticamente todas las técnicas clásicas de NLP**:

1. **Stop-words tienen sentido**: las top palabras ocupan ~30-50% del texto, casi todas funcionales. Filtrarlas reduce ruido sin perder contenido.
2. **TF-IDF es necesario**: castiga palabras frecuentes para compensar la dominancia de stop-words.
3. **El vocabulario crece sin parar**: la cola larga de Zipf contiene miles de hápax (palabras que aparecen 1 vez).
4. **Smoothing es obligatorio en n-gramas**: probabilidad cero rompería el modelo.

**Origen**: George Zipf (1949), *Human Behavior and the Principle of Least Effort*. Su intuición: el lenguaje balancea esfuerzo del hablante (palabras cortas, repetitivas) con especificidad del oyente (palabras distintivas).

---

## 5. La Ley de Heaps (celda 20)

> El vocabulario único de un texto crece como una raíz del número de tokens leídos: $V(N) \approx K \cdot N^\beta$ con $\beta \in [0.4, 0.6]$.

Si Zipf describe "cómo se distribuyen las frecuencias", Heaps describe **"cómo crece el vocabulario a medida que leemos más texto"**. Son las dos caras de la misma moneda estadística del lenguaje.

### Verificación empírica sobre Moby Dick

```python
x = []
y = []
words = set()
for i, word in enumerate(text1.tokens):
    words.add(word)
    x.append(i+1)
    y.append(len(words))

import matplotlib.pyplot as plt
plt.plot(x, y)
```

Gráfico esperado:

```
17000 ┤                                                ___________
       ┤                                    ___________
14000 ┤                            _________
       ┤                  __________
11000 ┤            _______
       ┤        ____
 8000 ┤     ____
       ┤  ___
 5000 ┤_/
       ┤
    0 ┤
       └─────────────────────────────────────────────────
        0       50k     100k    150k    200k    260k
                          Tokens leídos →
```

Forma **cóncava, creciente, nunca se aplana**. El primer tramo crece rápido (cada página agrega palabras nuevas). Hacia el final crece muy lento (la mayoría de palabras ya las viste).

### Por qué Heaps importa

1. **Justifica vocabularios "infinitos"** en producción: cualquier vocab fijo tendrá OOV en texto nuevo.
2. **Motiva subword tokenization** (BPE, WordPiece, SentencePiece): si las palabras únicas crecen indefinidamente, descomponé palabras en subunidades.
3. **Predice cuánto corpus necesitás**: para cubrir el 98% del vocabulario común inglés, necesitás ~100M tokens.

**Detalle empírico**: para Moby Dick (260k tokens, 17k vocab), si calculáramos $\beta$ por OLS log-log obtendríamos ~0.5. Para corpora más cortos como Genesis (44k tokens), $\beta$ puede ser ~0.6 — Heaps es aproximación, no ley exacta.

---

## 6. Comparativa cuantitativa entre corpora

Patrón típico:

```python
for text in [text1, text2, text3, text4, text5]:
    V = len(set(text.tokens))
    N = len(text.tokens)
    print(f"{text.name[:30]:30} N={N:6} V={V:5} TTR={V/N:.3f}")
```

Esperado:

| Corpus | N | V | TTR (V/N) |
|---|---|---|---|
| Moby Dick | 260,819 | ~17,200 | 0.066 |
| Sense and Sensibility | ~142,000 | ~6,200 | 0.044 |
| Genesis | ~44,000 | ~2,600 | 0.059 |
| Inaugural Address Corpus | ~150,000 | ~9,200 | 0.061 |
| Chat Corpus | ~45,000 | ~6,000 | 0.133 |

**TTR (Type-Token Ratio)** es medida cruda de riqueza léxica:
- Moby Dick: vocabulario náutico técnico amplio → 6.6%.
- Sense and Sensibility: lenguaje cotidiano repetitivo → 4.4%.
- Chat Corpus: cada usuario introduce palabras únicas → 13.3%.

**Limitación**: TTR depende del tamaño. Para comparación justa entre textos de tamaños distintos, usar MATTR (Moving Average TTR) o MTLD.

---

## 7. Aplicación a tu trabajo MDM-FHIR

Las leyes de Zipf y Heaps **aplican idénticamente** a texto clínico. Patrón típico:

| Tipo de corpus | $\beta$ Heaps | Implicación |
|---|---|---|
| Literario | ~0.45-0.50 | Vocabulario crece moderadamente |
| Clínico | ~0.50-0.55 | Vocabulario crece más rápido (nombres propios, fármacos, abreviaciones) |
| Web / redes sociales | ~0.55-0.65 | Vocabulario explota (slang, typos, nombres de marcas) |

**Para tu pipeline**: si vas a construir BoW sobre 10k reportes clínicos, esperá:
- Vocabulario crudo: ~30-50k palabras únicas.
- Con stemming + autocorrect + stop-words: ~10-15k.
- Con subword tokenization (BPE): ~3-5k tokens unitarios.

La práctica local del lab — scripts 10-11 en [clase_16/practica](https://github.com/) — verifica estas leyes empíricamente sobre 4 corpora clínicos (MEDDOCAN, Cantemist, PharmaCoNER, Quijote) para mostrar diferencias cross-domain.

---

## Lecturas

- Zipf (1949), *Human Behavior and the Principle of Least Effort* — el libro foundacional.
- Heaps (1978), *Information Retrieval: Computational and Theoretical Aspects* — la ley homónima.
- Mandelbrot (1953), *"An Informational Theory of the Statistical Structure of Language"* — derivación matemática.
- Baeza-Yates & Ribeiro-Neto (1999), *Modern Information Retrieval* — texto clásico.

Anterior: [Tokenización con NLTK](nltk-tokenizacion).
Siguiente: [Normalización: stop-words y stemming](nltk-normalizacion).
