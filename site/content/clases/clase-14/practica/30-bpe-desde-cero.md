---
title: "30 - BPE desde cero: el algoritmo que tokeniza GPT"
weight: 300
math: true
---

El [capitulo 29]({{< relref "clases/clase-14/practica/29-dpo-training-eval" >}}) mostro que SFT funciono (drift 40%→0%, repeat/qa al 100%) pero DPO no pudo mejorar claramente sobre el SFT — la accuracy bajo entre 5.5 y 24 puntos porcentuales en las cuatro tareas. La hipotesis mas probable: los tokens char-level son demasiado atomicos. Cada caracter es una unidad, el modelo no tiene nocion de palabras. "the king is dead" son 16 tokens char-level identicos en granularidad a los 4 tokens de "aaaa". No hay diferencia semiotica entre un caracter y otro — todos son unidades del mismo tamano sin importar si forman parte de una palabra comun o no.

BPE (Byte Pair Encoding) resuelve esto: aprende que "the " es un solo token, que "ing " es otro, que "not " es otro. Con BPE el modelo tiene semantica de palabras desde el primer token. Este es el primer capitulo de Camino 2.5, que en el cap 31 reentrenara el Mini-LLaMA desde cero con el tokenizer que construimos aqui.

---

## 1. El problema de vocabulario — el trade-off fundamental

Disenar un tokenizer es escoger un punto en un trade-off: unidades grandes comprimen bien pero no pueden manejar palabras no vistas; unidades pequenas manejan cualquier texto pero pierden estructura semantica.

Las tres estrategias principales:

**Char-level** (vocab ~65 para ASCII basico): cada token es un caracter. La ventaja es que el vocabulario es compacto y nunca aparecen palabras "fuera de vocabulario" — cualquier texto se puede tokenizar. La desventaja es que las secuencias son largas y el modelo no tiene acceso a unidades semanticas: "the" son tres tokens separados, "king" son cuatro. La red tiene que aprender internamente que `t-h-e` juntos significan el articulo definido ingles, en lugar de recibirlo ya agrupado.

**Word-level** (vocab 50k-100k): cada token es una palabra entera. Las secuencias son cortas — "the king is dead" son 4 tokens. Pero el vocabulario explota si el corpus tiene morfologia rica (castellano, aleman, turco), y cualquier forma no vista ("shakespeareano", "desintermediacion", una nueva palabra tecnica) no tiene ID y cae en el token especial `[UNK]`. El modelo no puede generalizar sobre raices y sufijos porque nunca los ve como unidades separadas.

**BPE** (vocab 1112 en nuestro caso): subwords. BPE aprende las unidades frecuentes del corpus y les asigna tokens propios. "the " (con espacio) es un token porque aparece muy frecuentemente. "ing " es otro token porque el sufijo de gerundio aparece en cientos de palabras. Las palabras raras se descomponen en subwords que si existen. Es el balance optimo entre cobertura y compresion.

| Estrategia | Vocab | "the king" | Problema principal |
|---|---|---|---|
| Char-level | 65 | 8 tokens | Sin semantica de palabras |
| Word-level | ~50k | 2 tokens | OOV masivo, vocab explota |
| BPE (ours) | 1112 | 3-4 tokens | Aprende las unidades relevantes |

GPT-2 usa BPE con vocab 50,257. GPT-4 usa tiktoken (una variante byte-level BPE) con vocab ~100k. Nuestro mini-BPE de 1112 tokens es el mismo principio, solo mucho mas chico.

---

## 2. El algoritmo BPE — paso a paso

El algoritmo original de Sennrich et al. 2016 es sorprendentemente simple. Parte de un vocabulario de caracteres unicos y, en cada iteracion, fusiona el par de tokens consecutivos mas frecuente en el corpus.

Ejemplo con corpus minimal: `"aaabdaaabac"`.

**Paso 0 — Vocabulario inicial:** extraer todos los caracteres unicos del corpus.

$$\text{vocab} = \{a, b, c, d\}$$

**Paso 0 — Tokenizacion inicial:** el corpus es una lista de tokens char-level.

$$\text{tokens} = [a, a, a, b, d, a, a, a, b, a, c]$$

**Paso 1 — Contar pares consecutivos:**

| Par | Count |
|---|---|
| (a, a) | 5 |
| (a, b) | 3 |
| (b, d) | 1 |
| (b, a) | 1 |
| (a, c) | 1 |

**Paso 1 — Mejor par:** `(a, a)` con count=5. Merge: `a + a → aa`. Nuevo token `aa` se agrega al vocab.

$$\text{vocab} = \{a, b, c, d, aa\}$$

**Paso 1 — Aplicar merge:** reemplazar todas las ocurrencias del par `(a, a)` con `aa`.

$$\text{tokens antes} = [a, a, a, b, d, a, a, a, b, a, c]$$
$$\text{tokens despues} = [aa, a, b, d, aa, a, b, a, c]$$

La secuencia paso de 11 tokens a 9. Los dos `aa` iniciales se fusionaron. El tercer `a` quedo suelto porque no habia otro `a` a su derecha inmediata (estaba `b`).

**Paso 2 — Repetir:** contar pares de nuevo sobre la secuencia nueva. Ahora `(aa, a)` aparece dos veces, `(a, b)` dos veces, etc. El proceso sigue por `num_merges` iteraciones. En cada paso la secuencia encoge levemente y el vocabulario crece en 1 token.

"Despues de N merges, los pares mas frecuentes del corpus se vuelven tokens propios. Para Shakespeare, 'the ' (con espacio) es ultra-frecuente y emerge como token en las primeras decenas de merges."

La implementacion en `_bpe.py` usa exactamente esta logica: inicializar el vocab con caracteres, luego iterar `num_merges` veces eligiendo el par de mayor frecuencia con `Counter` + `max`.

---

## 3. El script

`clase_14/practica/30_build_bpe.py`:

```python
"""30_build_bpe.py - Cap 30: BPE desde cero.

Entrena un BPETokenizer sobre Shakespeare + Quijote (~1MB bilingue).
Produce data/bpe_tokenizer.json con vocab ~1100 tokens.
"""
from pathlib import Path
from _bpe import BPETokenizer

NUM_MERGES = 1000

print("Cargando corpus bilingue (Shakespeare + Quijote)...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
print(f"Corpus: {len(corpus):,} chars total (usando primeros 50,000 para training)")

tok = BPETokenizer()
print(f"\nEntrenando BPE con {NUM_MERGES} merges...")
import time
t0 = time.time()
tok.train(corpus, num_merges=NUM_MERGES)
elapsed = time.time() - t0
print(f"Listo en {elapsed:.1f}s")

print(f"\nVocab size: {tok.vocab_size} tokens")
print(f"Merges aprendidos: {len(tok.merges)}")

# Verificar que \n es un token propio (importante para stop_token en generacion)
newline_id = tok.vocab.get("\n")
newline_status = f"id={newline_id} — OK" if newline_id is not None else "AUSENTE — problema"
print(f"\nToken '\\n' en vocab: {newline_status}")

# Ejemplos de tokenizacion
examples = [
    "the king is dead",
    "To be or not to be",
    "En un lugar de la Mancha",
    "INSTR: repeat 'a' three",
    "Q: who wrote Hamlet?",
]
print("\n=== Ejemplos de tokenizacion ===")
for ex in examples:
    ids = tok.encode(ex)
    tokens = [tok.id_to_token[i] for i in ids]
    print(f"  '{ex}'")
    print(f"    chars={len(ex)}  tokens={len(ids)}  ratio={len(ids)/len(ex):.2f}")
    print(f"    tokens: {tokens}")

# Guardar
Path("data").mkdir(exist_ok=True)
tok.save("data/bpe_tokenizer.json")
print(f"\nSaved -> data/bpe_tokenizer.json")

# Verificar round-trip
tok2 = BPETokenizer.load("data/bpe_tokenizer.json")
sample = "To be or not to be"
assert tok.encode(sample) == tok2.encode(sample), "round-trip fallo"
print("Round-trip verificado.")
print(f"\nVocab final: {tok.vocab_size} tokens")
```

La clase `BPETokenizer` vive en `clase_14/practica/_bpe.py`. El metodo `train` implementa el loop de merges descrito arriba: vocab inicial de chars unicos, luego `num_merges` iteraciones eligiendo el par mas frecuente con `Counter`. El metodo `encode` aplica los merges aprendidos en orden sobre el texto nuevo. El metodo `save/load` serializa el vocab y la lista de merges en JSON.

---

## 4. El output literal

```text
Cargando corpus bilingue (Shakespeare + Quijote)...
Corpus: 3,245,453 chars total (usando primeros 50,000 para training)

Entrenando BPE con 1000 merges...
Listo en 4.3s

Vocab size: 1112 tokens
Merges aprendidos: 1000

Token '\n' en vocab: id=0 — OK

=== Ejemplos de tokenizacion ===
  'the king is dead'
    chars=16  tokens=6  ratio=0.38
    tokens: ['the ', 'k', 'ing ', 'is ', 'de', 'ad']
  'To be or not to be'
    chars=18  tokens=7  ratio=0.39
    tokens: ['T', 'o ', 'be ', 'or ', 'not ', 'to ', 'be']
  'En un lugar de la Mancha'
    chars=24  tokens=14  ratio=0.58
    tokens: ['E', 'n ', 'un', ' ', 'l', 'u', 'gar', ' ', 'de ', 'la', ' ', 'M', 'an', 'cha']
  'INSTR: repeat 'a' three'
    chars=23  tokens=14  ratio=0.61
    tokens: ['IN', 'S', 'T', 'R', ': ', 're', 'pe', 'at ', "'", 'a', "'", ' th', 're', 'e']
  'Q: who wrote Hamlet?'
    chars=20  tokens=13  ratio=0.65
    tokens: ['Q', ':', ' w', 'ho ', 'w', 'ro', 't', 'e ', 'H', 'am', 'le', 't', '?']

Saved -> data/bpe_tokenizer.json
Round-trip verificado.

Vocab final: 1112 tokens
```

---

## 5. Lectura del output — lo interesante

**El ingles se comprime mejor que el espanol.** "the king is dead" obtiene ratio 0.38 — compresion del 62%, de 16 chars a 6 tokens. "En un lugar de la Mancha" obtiene ratio 0.58 — solo 42% de compresion, de 24 chars a 14 tokens. La diferencia no es accidental.

**Eso ES corpus bias en tokenizacion.** BPE se entrena sobre los primeros 50,000 chars del corpus — que son todos Shakespeare (ingles) porque en la concatenacion el ingles va primero. Los merges que el algoritmo aprende representan las frecuencias de pares en ese subconjunto ingles. "the " emerge como token porque el articulo definido ingles es el par mas frecuente. "ing " emerge porque el sufijo de gerundio es ubiquo en ingles. Las palabras castellanas, que tienen morfologia distinta (articulos, sufijos verbales, preposiciones distintas), se descomponen en subwords que el BPE ingles no conoce bien — de ahi los 14 tokens para "En un lugar de la Mancha".

**GPT-4 con tiktoken tiene el mismo problema a escala.** Tiktoken tokeniza ingles con ~4 chars/token. Castellano, frances, aleman con ~5-6 chars/token. Chino o japones en algunos casos con 1-2 chars/token porque los kanji concentran mucho significado. Nuestro tiny BPE exhibe el mismo fenomeno: el sesgo del corpus de entrenamiento se refleja directamente en la eficiencia de compresion por idioma. Si quisieramos un tokenizer balanceado EN-ES, habria que entrenar BPE sobre un corpus donde ambos idiomas tengan representacion proporcional — o poner el Quijote primero en la concatenacion para que los primeros 50,000 chars sean castellanos.

**El token `\n` existe como token propio (id=0).** Esto es critico para la generacion. En la funcion `generate_with_prompt` usamos `\n` como stop token: cuando el modelo genera una nueva linea, cortamos la secuencia. Si `\n` hubiera sido fusionado en un token multi-char (por ejemplo, si el par mas frecuente hubiera sido `\n` seguido de algo especifico), `tok.vocab.get("\n")` devolveria `None`, el `stop_id` seria invalido, y la generacion no cortaria nunca — seguiria hasta `max_new_tokens`. El hecho de que `\n` sea el primer token del vocab (id=0) confirma que es un caracter unico frecuente que el algoritmo no mergeo: aparece muy frecuente como separador de lineas pero no tiene un caracter especifico que lo siga siempre, entonces el algoritmo no tiene incentivo para fusionarlo con nada.

**Los formatos de instruccion ("INSTR:", "Q:") se tokenizan pobremente.** ratio 0.61-0.65 porque el corpus de entrenamiento BPE (Shakespeare + Quijote) no contiene estos prefijos. BPE no aprendio que "INSTR" es una unidad semantica — la descompone en `['IN', 'S', 'T', 'R']`. En produccion, tiktoken y similares se entrenan incluyendo el dataset de instrucciones para que los tokens de control sean eficientes. Aqui lo simplificamos deliberadamente para aislar la pedagogia: el tokenizer aprende del corpus literario, y vemos exactamente como ese sesgo se manifiesta.

**"To be or not to be" → 7 tokens con espacio incluido en el token.** Las palabras cortas ("be", "or", "not") cada una con su espacio adelante ("be ", "or ", "not ") emergen como tokens propios. Esto refleja una decision de implementacion: BPE sobre texto crudo (sin pre-tokenizacion por espacios) aprende que el espacio es parte del token. "be " es mas frecuente como unidad que los tres tokens `b`, `e`, ` ` por separado — el espacio siempre sigue a "be" en la mayoria de los contextos. Es mas eficiente y ademas preserva mejor la informacion de limite de palabra.

---

## 6. Por que `\n` como stop token importa

En `generate_with_prompt`, usamos `\n` como senal de fin de respuesta. El formato de instruccion es:

```text
INSTR: reverse 'cat'
RESP: tac
```

El modelo tiene que generar `tac\n` y parar. La nueva linea al final indica que termino la respuesta. Si `\n` no fuera un token propio del vocab BPE, no habria un ID de stop unico y la funcion de generacion tendria que buscar una secuencia de tokens que decodifica a `\n` — mucho mas complejo.

Con char-level (Caminos 1 y 2) esto era automatico: `\n` es siempre el caracter `\n`, siempre un token unico. Con BPE hay que verificarlo explicitamente. Por eso el script hace:

```python
newline_id = tok.vocab.get("\n")
newline_status = f"id={newline_id} — OK" if newline_id is not None else "AUSENTE — problema"
```

Si la salida fuera "AUSENTE — problema", habria que cambiar la estrategia de stop: pre-tokenizar por lineas, usar un token especial `[EOS]`, o buscar en la secuencia decodificada. Cualquiera de las tres opciones es mas compleja. El hecho de que `\n` sea id=0 simplifica toda la logica posterior de Camino 2.5.

El assert de round-trip cumple una funcion similar de verificacion: confirma que guardar y cargar el tokenizer produce exactamente los mismos IDs. Si `tok.encode(sample) != tok2.encode(sample)`, hay un bug en la serializacion de merges — probablemente un problema de orden (JSON puede reordenar claves en ciertos casos), que romperia la reproducibilidad del tokenizer entre sesiones.

---

## 7. Preguntas de verificacion

1. **¿Por que el espanol tiene ratio mas alto (peor compresion) que el ingles en nuestro BPE?**
   BPE se entrena sobre los primeros 50,000 chars del corpus concatenado, que son exclusivamente Shakespeare (ingles). Los 1000 merges aprendidos representan patrones de frecuencia del ingles. Las palabras castellanas tienen morfologia distinta — articulos, sufijos verbales, preposiciones — que no coinciden con los merges aprendidos, entonces se descomponen en subwords mas granulares y la compresion es peor.

2. **Si quisieras un tokenizer balanceado EN-ES, ¿que cambiarias?**
   Al menos tres opciones: (a) intercalar los dos corpus en el string de entrenamiento para que ambos idiomas esten representados en los 50,000 chars iniciales, (b) entrenar dos BPE separados y hacer union de vocabularios (tecnica usada en mBERT y XLM-R para tokenizacion multilingue), o (c) simplemente cambiar el orden de concatenacion para que el Quijote vaya primero.

3. **¿Que pasa si el token de stop (`\n`) no esta en el vocab BPE como token propio?**
   La funcion `generate_with_prompt` recibira `stop_id = None` (o un ID incorrecto). La generacion no tendra condicion de corte sobre nueva linea y seguira generando hasta `max_new_tokens`. En los experimentos del cap 31, esto produciria respuestas que no terminan o que terminan abruptamente por limite de tokens, no por contenido. Toda la eval exact-match depende de que el modelo genere exactamente la respuesta seguida de `\n` — sin stop correcto, la metrica colapsa.

---

## 8. Que viene

En el cap 31 entrenamos un Mini-LLaMA desde cero con este vocab de 1112 tokens. El pretrain es el mismo algoritmo de siempre — embedding table de tamano 1112 en vez de 65, mismo Transformer, mismo cross-entropy. Solo cambia el tokenizer que convierte texto a IDs y de vuelta. La hipotesis es que con unidades semanticas mas ricas (palabras completas en vez de chars), el modelo aprendera estructuras linguisticas mas rapidamente — y el SFT posterior deberia funcionar mejor.

Volver al [hub de practica](..) o a la [Clase 14](../..).
