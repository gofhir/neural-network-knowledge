# Análisis interno — Cañete et al. (2020) "Spanish Pre-trained BERT Model and Evaluation Data" (BETO)

> Documento complementario al material público del site sobre BETO (`papers/beto-canete-2020.md`, `fundamentos/beto-spanish-bert.md`) y al notebook del Lab 20 (celdas 25-33). Aquí se profundiza en el contexto que motivó un BERT monolingüe en español, las decisiones de corpus y tokenización, los detalles de pre-training que el paper aclara solo en una página, el benchmark GLUES, las versiones liberadas, las limitaciones y la evolución posterior de la familia BERT-español en LATAM. Se trata además con cuidado especial porque el paper proviene del **mismo departamento que dicta el Diplomado** (DCC UChile / IMFD), con Jorge Pérez como autor senior — figura referencial del CENIA y del ecosistema chileno de IA.

- **Paper**: Cañete, Chaperon, Fuentes, Ho, Kang, Pérez. *Spanish Pre-trained BERT Model and Evaluation Data*. Workshop paper at **PML4DC, ICLR 2020** (Practical ML for Developing Countries). 10 páginas, sin proceedings formales — es un workshop paper, no un paper de track principal.
- **PDF local**: [`Canete-BETO-2020.pdf`](./Canete-BETO-2020.pdf)
- **Código y datos**:
  - Repo principal: `https://github.com/dccuchile/beto`
  - Corpus: `https://github.com/josecannete/spanish-corpora`
  - Benchmark GLUES: `https://github.com/dccuchile/glues`
- **Modelos en Hugging Face** (organización `dccuchile`):
  - `dccuchile/bert-base-spanish-wwm-cased` — versión cased, la más usada en producción
  - `dccuchile/bert-base-spanish-wwm-uncased` — versión uncased
  - Posteriores (no del paper original): `dccuchile/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`, `dccuchile/distilbert-base-spanish-uncased`, modelos NER y POS específicos
- **Licencia**: CC BY 4.0 para modelos y datos.

---

## 1. Contexto histórico: el vacío de modelos pre-entrenados en español (2019-2020)

A finales de 2019, el panorama de transfer learning en NLP era marcadamente **anglocéntrico**. Tras la explosión de BERT (Devlin et al., octubre 2018), el ecosistema producido durante 2019 — RoBERTa (Liu et al.), ALBERT (Lan et al.), XLNet (Yang et al.), SpanBERT (Joshi et al.), DistilBERT (Sanh et al.), ELECTRA (Clark et al.) — operaba casi exclusivamente sobre corpus en inglés. La razón no era técnica sino infraestructural: los corpora masivos y bien curados estaban en inglés (BooksCorpus, OpenWebText, C4 venido después), los benchmarks de evaluación eran en inglés (GLUE, SuperGLUE, SQuAD), y los grupos con TPUs disponibles para experimentar tenían los datos y las tareas en inglés a la mano.

Google liberó dos paliativos al monolingüismo en 2018:

1. **BERT-Chinese** — un modelo BERT-base pre-entrenado sobre Wikipedia en chino con tokenización por carácter, demostrando que la arquitectura no es específica al inglés.
2. **mBERT (Multilingual BERT)** — un único modelo BERT-base entrenado simultáneamente sobre Wikipedia de **104 idiomas**, con un vocabulario WordPiece compartido de **110,000 tokens** y up-sampling exponencial (factor 0.7) de idiomas de bajo recurso para compensar el sesgo hacia inglés. El argumento era que un modelo "todo en uno" podía servir a la comunidad global de NLP sin necesitar uno por idioma.

mBERT funcionó mejor de lo esperado, especialmente para tareas cross-lingüe (Wu & Dredze 2019 — el paper "Beto, Bentz, Becas: The surprising cross-lingual effectiveness of BERT" — ironía del nombre dado que comparte con el modelo chileno). Pero quedaron dos limitaciones estructurales:

### 1.1 Sub-representación del español en mBERT

El vocabulario de 110K tokens debe cubrir 104 idiomas. Aun con up-sampling, los idiomas con más datos dominan el vocabulario. Una estimación razonable es que el español tiene menos de **5,000 tokens propios** en el vocab de mBERT, comparado con los aproximadamente **30,000 tokens en inglés**. Consecuencia directa: palabras españolas comunes se fragmentan en muchos más subwords que sus equivalentes en inglés. El sufijo flexivo español (`-ando`, `-iendo`, `-aríamos`, `-amientos`) tiene poca representación dedicada y se descompone en piezas de propósito general que el modelo nunca aprendió a tratar como morfemas coherentes.

### 1.2 Corpus de pre-training desbalanceado

El corpus de mBERT proviene de Wikipedia. Wikipedia en español tenía en 2019 aproximadamente **400-500 millones de palabras** (~1.5M de artículos), versus 2.5 mil millones en inglés. Incluso con up-sampling, el modelo ve un orden de magnitud menos español del que recibió un BERT inglés.

### 1.3 Estado del arte previo en NLP español

Antes de BETO, los recursos pre-entrenados en español eran principalmente:

| Recurso | Año | Tipo | Limitación |
|---|---|---|---|
| **Spanish FastText** (Bojanowski et al.) | 2017 | Embeddings estáticos con n-gramas de caracteres | Una representación por palabra (sin contexto) |
| **Spanish Billion Words Embeddings** (Cardellino) | 2016 | word2vec sobre ~1.5B palabras | Estáticos, vocabulario fijo |
| **GloVe español** (varios reimplementaciones) | 2014-2018 | Co-ocurrencia matricial | Estáticos |
| **Spanish ELMo** (varios) | 2018-2019 | BiLSTM contextualizado | Más lento, bidireccionalidad shallow |
| **mBERT** (Google) | 2018 | Transformer encoder, 104 idiomas | Sub-representación de español, vocab compartido |

El salto cualitativo que ofrecía BERT — bidireccionalidad profunda con representaciones contextuales transferibles vía fine-tuning a tareas downstream — estaba accesible solo de forma diluida vía mBERT, o no estaba accesible en absoluto para tareas que exigen sensibilidad fina al español (NER médico, clasificación legal, análisis de sentimiento sobre textos formales).

### 1.4 Demanda concreta desde LATAM

El paper se publica en abril 2020 en PML4DC (Practical Machine Learning for Developing Countries), un workshop de ICLR enfocado explícitamente en problemas y comunidades fuera del eje anglo. Esta elección de venue no es accidental: el grupo de Pérez en DCC UChile (con Cañete liderando el trabajo experimental) entendía que un BERT español tenía una **base de usuarios real esperando**:

- Hospitales y aseguradoras en LATAM con historiales clínicos en español que requerían NER médico, normalización de medicamentos, codificación CIE-10/SNOMED.
- Bancos y fintech con sistemas KYC, anti-fraude, scoring crediticio basados en texto libre.
- Sectores legales y gubernamentales con grandes corpus de jurisprudencia, regulaciones, contratos.
- Industrias de servicio al cliente con sistemas conversacionales, ticketing, análisis de sentimiento sobre opiniones.

mBERT era una opción viable pero subóptima para todos estos casos. Faltaba el modelo monolingüe que ya tenían franceses (CamemBERT, FlauBERT), holandeses (BERTje, RobBERT), italianos (AlBERTo), portugueses (Souza et al. 2019), rusos (RuBERT). El paper lo dice explícitamente en la introducción: "Despite Spanish being widely spoken (much more than the previously mentioned languages) finding resources to train or evaluate Spanish language models is not an easy task."

BETO llena ese hueco. Y lo hace con un timing particular: justo antes de la explosión de modelos generativos (GPT-3 saldría en junio 2020, tres meses después), en el último momento histórico donde un "BERT para X-idioma" todavía era una contribución significativa por sí misma.

### 1.5 Otros BERTs monolingües no-ingleses como referencia

| Modelo | Idioma | Año | Autores | Corpus aprox. |
|---|---|---|---|---|
| BERT-Chinese (Devlin) | Chino | 2018 | Google | Wikipedia ZH |
| RuBERT (Kuratov & Arkhipov) | Ruso | 2019 | DeepPavlov | Wikipedia RU + News |
| CamemBERT (Martin et al.) | Francés | 2019 | INRIA + Facebook | OSCAR FR (~138GB) |
| FlauBERT (Le et al.) | Francés | 2019 | LIG + LISN | 71GB francés |
| BERTje (de Vries et al.) | Holandés | 2019 | Groningen | ~12GB holandés |
| RobBERT (Delobelle et al.) | Holandés | 2020 | KU Leuven | OSCAR NL |
| AlBERTo (Polignano et al.) | Italiano | 2019 | Bari | TWITA (Twitter italiano) |
| FinBERT (Virtanen et al.) | Finés | 2019 | Turku | Internet FI |
| BERTimbau / "Souza BERT-PT" | Portugués (BR) | 2019 | NeuralMind | brWaC + Wikipedia PT |
| **BETO (Cañete et al.)** | **Español** | **2020** | **DCC UChile** | **~3B palabras** |

BETO llega un poco tarde — francés, holandés, italiano, portugués y ruso ya tenían su BERT monolingüe. Pero llega como la primera contribución abierta y reproducible para español, con corpus liberado, código liberado, modelos liberados en Hugging Face, y benchmark de evaluación liberado. La fecha de release marca el inicio del ecosistema español-NLP moderno.

---

## 2. Autores y procedencia institucional

El paper tiene **seis autores**, divididos entre dos departamentos de la Universidad de Chile:

| Autor | Departamento | Rol en el paper |
|---|---|---|
| **José Cañete** | DCC (Computer Science) | Lead author, trabajo experimental. Nota a pie: "Work partially performed while at Adereso" |
| **Gabriel Chaperon** | DCC | Co-author, ingeniería |
| **Rodrigo Fuentes** | DCC | Co-author, ingeniería |
| **Jou-Hui Ho** | EE (Electrical Engineering) | Co-author (estudiante undergrad, marcado `ug.uchile.cl`) |
| **Hojin Kang** | EE | Co-author (estudiante undergrad) |
| **Jorge Pérez** | DCC + IMFD | **Senior author**, asesor académico, conexión con Millennium Institute for Foundational Research on Data |

Es importante destacar que esta no es la lista habitual que circula en Hugging Face Hub ni en redes sociales. Muchas referencias informales mencionan a "Cañete y Pérez" solamente, omitiendo a los cuatro co-autores intermedios. El paper completo es trabajo de un equipo de seis personas.

### 2.1 José Cañete

PhD candidate del DCC UChile al momento del paper. Posteriormente continuó trabajo en BETO, ALBERTO español, y modelos derivados (DistilBETO, RoBERTa-base-spanish). Mantiene el repositorio `josecannete/spanish-corpora` con el dataset de pre-training liberado para reuso. Es la persona-puerta-de-entrada al ecosistema BETO en HuggingFace.

### 2.2 Jorge Pérez

Profesor del DCC UChile, investigador principal del **Millennium Institute for Foundational Research on Data (IMFD)**. Su trayectoria académica se centró históricamente en **bases de datos, lógica computacional y teoría de grafos**, con contribuciones notables al área de bases de datos en grafos (SPARQL, RDF, foundations of graph query languages). Su giro hacia NLP y deep learning en 2018-2019 fue parte del movimiento más amplio del DCC UChile hacia el ML aplicado.

Posteriormente, Pérez se convirtió en figura visible del **Centro Nacional de Inteligencia Artificial (CENIA)**, el centro de IA financiado por ANID en Chile que aglutina investigadores de DCC UChile, PUC, Universidad de Concepción y otras instituciones. CENIA es uno de los puentes naturales entre el ecosistema BETO y el Diplomado IA UC: muchos profesores y profesionales del entorno comparten redes, talleres y eventos con el equipo de Pérez.

### 2.3 Conexión con el Diplomado IA UC

La PUC (Pontificia Universidad Católica de Chile, donde se dicta el Diplomado IA UC) y la Universidad de Chile (DCC, donde se produjo BETO) son las dos universidades chilenas con programas históricos en computación y IA. Comparten:

- Membresía conjunta en CENIA.
- Proyectos colaborativos vía IMFD (Pérez es del IMFD, varios PIs del IMFD están en PUC).
- Eventos comunes (Jornadas Chilenas de Computación, Encuentro Chileno de NLP).
- Movilidad de estudiantes y profesores entre ambos campus (a 15 min en metro).

Para los estudiantes del Diplomado IA UC, BETO es **el modelo NLP español más cercano institucionalmente**. Estudiarlo en la clase 20 es estudiar trabajo hecho a 8 km del aula, por colegas que se cruzan con los profesores del Diplomado en seminarios. El paper merece, por esa razón, una lectura más cuidadosa de la que se le daría a un paper genérico de un grupo internacional remoto.

### 2.4 Adereso

La nota a pie 1 indica que parte del trabajo de Cañete se hizo en **Adereso**, una startup chilena de plataformas conversacionales para atención al cliente (chatbots empresariales, omnicanalidad). Adereso financió compute para el entrenamiento del modelo uncased. Este es un dato relevante de transferencia academia-industria — la versión `uncased` que mucha gente usa para análisis de sentimiento en redes sociales fue posible por sponsorship corporativo chileno.

---

## 3. Arquitectura: idéntica a BERT-base (con una sutileza)

El paper declara (Sección 3): "We trained a model similar in size to a BERT-Base model. Our model has 12 self-attention layers with **16 attention-heads each**, using **1024 as hidden size**. In total our model has 110M parameters."

Aquí hay un detalle anómalo que conviene precisar. **El BERT-base estándar de Devlin et al. (2018) tiene 12 capas, 12 heads, y hidden size 768**, no 16 heads ni 1024 hidden. La configuración descrita en el paper de BETO (12L, 16H, 1024 dim, 110M params) **no cuadra aritméticamente con BERT-base ni con BERT-large**:

| Modelo | Capas $L$ | Hidden $H$ | Heads $A$ | FFN | Params estimados |
|---|---|---|---|---|---|
| BERT-base (Devlin) | 12 | 768 | 12 | 3072 | 110M |
| BERT-large (Devlin) | 24 | 1024 | 16 | 4096 | 340M |
| BETO según paper | 12 | 1024 | 16 | ? | "110M" — pero matemáticamente serían ~160M |
| BETO en HuggingFace `config.json` (real) | 12 | 768 | 12 | 3072 | 109M |

La inconsistencia más probable es un **error de escritura en el paper**: BETO replica exactamente BERT-base (12L, 768H, 12A, FFN 3072, ~110M params) según se verifica al inspeccionar el `config.json` de los modelos en Hugging Face (`dccuchile/bert-base-spanish-wwm-cased`). El número 1024 e 16 heads se cuelan probablemente porque los autores compararon notas con BERT-large mientras escribían. Para fines de implementación y reproducción, **vale la configuración del config.json en HF, no la del paper**.

### 3.1 Por qué no inventaron arquitectura nueva

El paper no introduce ninguna modificación arquitectónica respecto a BERT-base. Mismo encoder Transformer, mismo MLM + NSP, misma cabeza de pre-training, mismas convenciones de tokens especiales (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`). El aporte del paper no está en la arquitectura, sino en:

1. El **corpus** en español (compilado, curado, liberado).
2. El **vocabulario** dedicado al español (31K subwords entrenados solo en datos españoles).
3. El **régimen de entrenamiento** (Dynamic Masking 10x heredado de RoBERTa, Whole-Word Masking, batch grande, 2M pasos).
4. El **benchmark GLUES** para evaluación en español.

Esta decisión de no innovar arquitectónicamente es deliberada y razonable: el objetivo es producir un BERT-español usable inmediatamente como reemplazo drop-in de mBERT en cualquier pipeline existente. Cualquier modificación arquitectónica habría introducido fricción de adopción.

### 3.2 Importancia del weight-sharing con BERT estándar

Como BETO replica exactamente la arquitectura BERT-base, cualquier código que cargue BERT en PyTorch o TensorFlow funciona con BETO sin cambios:

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
```

**No existe una clase `BetoModel` en HuggingFace**. Esto no es accidente: si BETO hubiera modificado la arquitectura, habría requerido una clase separada. Al mantener compatibilidad total, los autores facilitaron que cualquiera con experiencia previa en BERT pudiera adoptar BETO sin curva de aprendizaje. Este patrón "arquitectura oficial + pesos comunitarios" es el modelo dominante en HuggingFace para todos los BERTs monolingües (CamemBERT es la excepción notable, con clase propia `CamembertModel`).

---

## 4. Corpus de pre-training: ~3 mil millones de palabras

El corpus es la contribución más material del paper. Se compone de:

### 4.1 Fuentes

| Fuente | Tipo | Volumen aprox. | Característica |
|---|---|---|---|
| **Wikipedia español** | Enciclopedia | ~500M palabras | Texto formal, neutro, multitemático |
| **OPUS — OpenSubtitles** | Subtítulos de cine/TV | ~1B palabras | Diálogo informal, oral, frases cortas |
| **OPUS — ParaCrawl** | Web crawl multilingüe | Variable | Texto web general, ruido alto |
| **OPUS — UN (United Nations)** | Documentos oficiales ONU | Cientos de M | Formal, jurídico-administrativo, técnico |
| **OPUS — EUBookshop** | Publicaciones UE | Cientos de M | Formal, regulatorio europeo |
| **OPUS — TED Talks** | Transcripciones de charlas TED | ~10M palabras | Oral preparado, divulgativo |
| **OPUS — DOGC** | Diari Oficial de la Generalitat de Catalunya (texto en español) | ~5-10M | Jurídico-administrativo catalán/español |
| **OPUS — News (varios subcorpus)** | Notas periodísticas | Cientos de M | Periodístico |

### 4.2 OPUS Project

OPUS (Tiedemann, 2012) es un repositorio académico abierto de **corpus paralelos multilingües**, mantenido en Helsinki. Originalmente diseñado para investigación en traducción automática estadística (los pares paralelos eran su producto principal), también almacena las versiones monolingües desalineadas. El equipo de BETO usó OPUS como aggregator one-stop-shop: en vez de crawlear ellos mismos diferentes fuentes, descargaron de OPUS los subcorpus que ya estaban curados, filtrados por idioma, y disponibles.

Esta decisión tiene una consecuencia importante: el corpus de BETO es **principalmente texto formal/escrito** (Wikipedia, ONU, UE, news) con una porción significativa de **diálogo cinematográfico** (OpenSubtitles). Lo que **no** está bien representado:

- Textos de redes sociales (Twitter, Facebook, foros).
- Textos médicos clínicos (que requieren consentimientos y de-identificación).
- Textos jurídicos LATAM (jurisprudencia local). El DOGC es catalán/español peninsular.
- Variantes regionales informales (rioplatense, chileno coloquial, mexicano oral).

### 4.3 Volumen total: 3 mil millones de palabras

El paper declara "about 3 billion words". Para comparación:

| Modelo | Corpus pre-training | Palabras |
|---|---|---|
| BERT (Devlin) | BooksCorpus + Wikipedia EN | 3.3B |
| RoBERTa (Liu) | + CC-News + OpenWebText + Stories | ~30B |
| **BETO (Cañete)** | Wikipedia ES + OPUS ES | **~3B** |
| CamemBERT (Martin) | OSCAR FR | ~138GB ~32B palabras |
| XLM-R (Conneau) | CommonCrawl 100 idiomas | 2.5TB |

BETO entrena con un corpus comparable en tamaño al BERT original — un orden de magnitud menos que RoBERTa, dos órdenes menos que XLM-R. Esto coloca al modelo en una **posición de moderate-data BERT**, con suficiente texto para aprender patrones generales del español pero insuficiente para emular el comportamiento de modelos modernos con corpora masivos.

### 4.4 Comparación implícita con mBERT

mBERT entrena con Wikipedia de 104 idiomas. La porción española es aproximadamente **400-500M palabras** (Wikipedia ES de 2018). BETO entrena con ~6× más texto **dedicado** al español, sin tener que repartir capacidad del modelo entre 104 idiomas. La superioridad de BETO sobre mBERT en las tareas en español del paper (Sección 5) es consistente con esta diferencia de datos y especialización.

### 4.5 Preprocesamiento

El paper da pocos detalles. Lo deducible:

- Deduplicación de líneas/párrafos repetidos (estándar en corpus building).
- Filtrado de idioma (mantener solo texto que el detector identifica como español, descartar mezclas).
- Normalización de caracteres (unicode NFC, probablemente).
- Sin filtrado de toxicidad explícito (el campo no era foco antes de 2020).
- Sin filtrado por calidad/perplejidad (el campo no había popularizado el approach todavía — BERTIN español lo implementaría más tarde, 2021).

Cardellino (2016) había compilado previamente el "Spanish Billion Words Corpus and Embeddings" (~1.5B palabras), que BETO **actualiza y amplía** según declara explícitamente el paper.

---

## 5. Tokenizador: SentencePiece BPE, 31K subwords

Aquí el paper introduce una diferencia importante respecto a BERT original que conviene precisar.

### 5.1 SentencePiece BPE, no WordPiece

El paper dice (Sección 3): "we constructed a vocabulary of 31K subwords using the **byte pair encoding** algorithm provided by the **SentencePiece** library (Kudo & Richardson, 2018). We added 1K place-holder tokens for later use which gave us a vocabulary of **32K tokens**."

Esto difiere de BERT original, que usaba **WordPiece**:

| Algoritmo | Criterio de merge | Marcador de subword | Librería típica |
|---|---|---|---|
| WordPiece (Schuster & Nakajima 2012, Wu et al. 2016) | Maximizar likelihood unigram LM (equivalente a $P(ab)/P(a)P(b)$) | `##` continuación (BERT) | Google internal, `tokenizers` HF |
| BPE (Sennrich et al. 2016) | Fusionar par más frecuente | `Ġ` inicio (GPT, RoBERTa) | `subword-nmt`, `tokenizers` HF |
| SentencePiece (Kudo & Richardson 2018) | Wrapper de BPE o Unigram LM, trata el texto como secuencia bruta sin pretokenización | `▁` inicio de palabra (espacio) | `sentencepiece` |

SentencePiece tiene la propiedad útil de **no requerir pre-tokenización por espacios**: trata el input como una secuencia de bytes/caracteres y aprende los merges sobre el texto crudo. Esto es ventajoso para idiomas sin segmentación por espacios (chino, japonés), y conveniente como práctica uniforme.

Sin embargo, hay un detalle de implementación que la gente confunde frecuentemente: aunque BETO entrena el vocabulario con SentencePiece+BPE, **los modelos en HuggingFace se cargan con `BertTokenizer` que usa convención WordPiece (`##` como marcador de continuación)**. Esto sugiere que el vocabulario producido por SentencePiece se convirtió/exportó al formato WordPiece antes del release, para mantener compatibilidad drop-in con la clase `BertModel`. El usuario final no necesita conocer esta complejidad — simplemente carga `BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")` y obtiene tokens con `##` continuación, como cualquier BERT.

### 5.2 Tamaño del vocabulario: 31K (+1K placeholders)

| Comparación | Vocab size |
|---|---|
| BERT-base inglés (uncased) | 30,522 |
| BERT-base inglés (cased) | 28,996 |
| **BETO (cased y uncased)** | **31,002 + 1,002 placeholders = 32,004** |
| mBERT | 119,547 (compartido entre 104 idiomas) |
| XLM-R | 250,002 (compartido entre 100 idiomas) |

Los 1K tokens placeholder son una decisión interesante: dejan espacio para que usuarios downstream puedan agregar tokens específicos del dominio (tags médicos, entidades empresariales, símbolos químicos) sin tener que redimensionar la matriz de embeddings.

### 5.3 Cobertura morfológica del español

El español tiene morfología flexiva rica (~50 formas verbales por verbo regular, género y número en sustantivos/adjetivos, derivación productiva). Un vocabulario dedicado al español puede dar tokens completos a sufijos productivos como:

- `##ando`, `##iendo` (gerundios)
- `##aría`, `##aríamos`, `##arían` (condicional)
- `##amiento`, `##miento`, `##ación` (sustantivación)
- `##itos`, `##itas`, `##illos`, `##illas` (diminutivos)
- `##able`, `##ible` (deónticos)
- `##mente` (adverbialización)

En mBERT estos sufijos compiten por espacio con sufijos análogos de otros idiomas y rara vez obtienen una entrada dedicada, llevando a fragmentación. Una palabra como **`comerían`** (condicional 3ra persona plural de "comer"):

- **BETO**: probable tokenización en 1-2 piezas (`comerían` o `com ##erían`).
- **mBERT**: tokenización típica en 3-4 piezas (`com ##er ##ía ##n` o `co ##mer ##ían` — depende de los merges aprendidos).

Esta diferencia de fragmentación tiene impacto cuantificable en tareas downstream: cuantos más subwords se generan por palabra, más capacidad del modelo se desperdicia en reagregar pedazos, y menos contexto efectivo cabe en los 512 tokens del input. Para textos clínicos largos en español, BETO permite procesar 1.3-1.5× más contenido por inferencia que mBERT.

### 5.4 Tokens especiales

BETO mantiene exactamente los tokens especiales de BERT original:

| Token | ID | Función |
|---|---|---|
| `[PAD]` | 0 | Padding |
| `[UNK]` | 1 | Out-of-vocabulary |
| `[CLS]` | 2 | Inicio de secuencia, clasificación |
| `[SEP]` | 3 | Separador de segmentos / fin de secuencia |
| `[MASK]` | 4 | Token enmascarado en MLM |

A diferencia de RoBERTa (que usa `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>` siguiendo convención fairseq), BETO usa exactamente la misma convención que BERT. Esto refuerza la compatibilidad drop-in con código existente.

### 5.5 Versiones cased y uncased

Igual que BERT original:

- **cased**: preserva mayúsculas y diacríticos (acentos, ñ). Vocab incluye versiones mayúsculas y minúsculas separadas.
- **uncased**: lowercase + remoción de acentos vía `unicodedata.normalize("NFD")` + filtrado de combining marks. **Importante**: la versión uncased de BETO también remueve acentos, lo cual es problemático para el español porque pares como `papá`/`papa`, `sí`/`si`, `dé`/`de`, `mí`/`mi`, `más`/`mas` se colapsan a un mismo token. Para tareas donde estos pares contrastivos importan, hay que usar la versión cased.

---

## 6. Whole-Word Masking (WWM)

### 6.1 Concepto y origen

Whole-Word Masking es una variación de la regla 80/10/10 estándar de BERT, propuesta originalmente por Cui et al. (2019) para chino y posteriormente integrada en una **actualización del checkpoint original de BERT** liberada en mayo 2019.

La regla estándar de BERT enmascara **subwords individuales**. Si una palabra se fragmenta en varias piezas WordPiece, cada pieza tiene 15% de probabilidad **independiente** de ser enmascarada. Consecuencia: una palabra puede tener algunas piezas enmascaradas y otras no.

Ejemplo con la palabra "comerían" tokenizada como `com ##er ##ían`:

| Token | Decisión MLM estándar |
|---|---|
| `com` | 15% prob de mask |
| `##er` | 15% prob de mask |
| `##ían` | 15% prob de mask |

Si solo `##er` resulta enmascarado, el modelo predice `er` viendo `com [MASK] ##ían` — una tarea trivial porque las piezas adyacentes acotan fuertemente la respuesta.

**Whole-Word Masking** cambia la unidad de masking: cuando una pieza es seleccionada para mask, **todas las piezas de la misma palabra original** también se enmascaran:

| Token | Decisión MLM con WWM |
|---|---|
| `com` | 15% prob de mask palabra completa |
| `##er` | (heredado de la decisión sobre `com`) |
| `##ían` | (heredado de la decisión sobre `com`) |

Si `com` se elige, se enmascara también `##er` y `##ían`. El modelo predice la palabra completa "comerían" viendo el contexto a su izquierda y derecha. Este escenario es **mucho más difícil** y fuerza al modelo a aprender representaciones semánticas, no atajos morfo-superficiales.

### 6.2 Por qué WWM importa más en español que en inglés

En inglés, la mayoría de palabras se tokenizan como una sola pieza WordPiece. La fragmentación promedio es ~1.1 piezas/palabra (palabras comunes raramente se fragmentan). Por tanto, WWM y masking estándar coinciden en la mayoría de los casos.

En español, la morfología flexiva genera muchas más fragmentaciones. La fragmentación promedio en BETO es del orden de ~1.3-1.5 piezas/palabra. WWM **cambia significativamente** la dificultad del objetivo MLM: en vez de predecir fragmentos morfológicos en contextos triviales, el modelo predice palabras completas en contextos abiertos.

El paper de BETO destaca explícitamente WWM como una de las tres mejoras técnicas que diferencian su régimen del BERT original (Sección 3, junto con Dynamic Masking 10x y batch size grande). El sufijo `-wwm-` en los nombres de los modelos (`bert-base-spanish-wwm-cased`) lo señala públicamente.

### 6.3 Detalle: ¿se mantiene el 15% global?

Una pregunta natural: si WWM enmascara grupos enteros de subwords, ¿se mantiene la proporción del 15% de subwords enmascarados en total?

La respuesta práctica (heredada de Cui et al. 2019): se selecciona un **15% de palabras completas** para enmascarar, no un 15% de subwords. En términos de subwords totales enmascarados, el porcentaje sube ligeramente (porque las palabras seleccionadas pueden ser de 2-4 subwords), pero la convención del paper se mantiene en "15% del input".

### 6.4 Dynamic Masking 10x

El paper también declara usar **Dynamic Masking** (heredado de RoBERTa, Liu et al. 2019), específicamente "10x" — cada oración del corpus tiene **10 patrones de masking diferentes** generados a priori, en vez de uno solo (static masking de BERT original).

La diferencia con la dynamic masking pura de RoBERTa: RoBERTa genera el masking al vuelo en cada batch, dando infinitas variantes. BETO genera 10 variantes pre-computadas. Es un compromiso entre el static masking (1 variante, BERT original) y el dynamic masking puro (infinitas variantes, RoBERTa). 10x es suficiente para que la mayoría de epochs vean patrones distintos sin sobrecargar el preprocesamiento.

---

## 7. Régimen de entrenamiento: detalles del paper

### 7.1 Hyperparams principales

| Parámetro | Valor BETO | Valor BERT original |
|---|---|---|
| Pasos totales | **2,000,000** | 1,000,000 |
| Optimizador | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam |
| Learning rate | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| LR warmup | 10,000 pasos | 10,000 pasos |
| LR schedule | Linear decay después de warmup | Linear decay |
| Weight decay | 0.01 | 0.01 |
| Dropout | 0.1 | 0.1 |
| **Fase 1** (primeros 900K pasos) | batch **2048**, seq len **128** | batch 256, seq len 128 |
| **Fase 2** (resto, 1.1M pasos) | batch **256**, seq len **512** | batch 256, seq len 512 (últimos 100K) |
| Dynamic Masking | 10x | Static (1x) |
| Whole-Word Masking | Sí | Solo en actualización posterior, no en paper original |

### 7.2 Schedule de dos fases (You et al. 2019)

El paper cita explícitamente a **You et al. 2019** ("Large batch optimization for deep learning: training BERT in 76 minutes") para justificar el schedule. La idea: entrenar primero con secuencias cortas (128) y batches grandes (2048) para que el modelo aprenda los patrones generales rápidamente, luego refinar con secuencias largas (512) y batches más pequeños (256) para que las posiciones largas y dependencias de largo alcance se ajusten.

Cálculo de cómputo total:

- Fase 1: 900K pasos × 2048 × 128 = **236B tokens** procesados.
- Fase 2: 1.1M pasos × 256 × 512 = **144B tokens** procesados.
- **Total: ~380B tokens**.

Para un corpus de 3B palabras (~5B tokens BPE), esto son **~76 épocas** sobre el corpus. Considerablemente más que las ~40 épocas de BERT original.

### 7.3 Batch grande (2048) en fase 1

Entrenar con batch 2048 a learning rate $10^{-4}$ requiere optimización cuidadosa. You et al. (2019) introdujeron **LAMB** (Layer-wise Adaptive Moments optimizer for Batch training) precisamente para hacer estable el entrenamiento de BERT con batches enormes. El paper de BETO no especifica si usaron LAMB o Adam puro — la mención de "Adam" en la sección de fine-tuning sugiere Adam, pero la cita a You et al. 2019 sugiere que el régimen de batch grande está inspirado en LAMB. Es un detalle de implementación que sería resoluble inspeccionando el código del repo.

### 7.4 2M pasos: doble que BERT original

BERT original entrena 1M pasos. BETO entrena 2M. Esta es una decisión inspirada en RoBERTa, que mostró que más pasos de pre-training siguen mejorando performance downstream incluso cuando el modelo ya converge en la loss de pre-training. Para un corpus de tamaño moderado (3B palabras), pasar más tiempo extrayendo señal del mismo texto es eficiente uso de compute.

---

## 8. Hardware: TPU v3-8 preemptible vía Google TFRC

El paper especifica: "All the pre-training was done using **Google's preemptible TPU v3-8**."

### 8.1 TPU v3-8 — especificaciones

| Componente | Especificación |
|---|---|
| Chips por TPU device | 8 (un "TPU v3 pod slice") |
| HBM por chip | 16 GB |
| HBM total por device | 128 GB |
| Throughput por chip | 123 TFLOPS bfloat16 |
| Throughput por device | 420 TFLOPS bfloat16 |
| Interconexión | 2D torus, 656 GB/s/chip |

Una TPU v3-8 era en 2019-2020 el escalón de entrada del programa Cloud TPU de Google. Para reference, BERT-large original fue entrenado en **16 TPUs** (TPUv2, predecesoras). BETO usa **1 TPU v3-8** — un orden de magnitud menos compute paralelo, compensado con más tiempo wall-clock (varios días/semanas).

### 8.2 Modo preemptible

**Preemptible** significa que Google puede terminar el job en cualquier momento (si necesita la TPU para clientes que pagan tarifa completa), con costo significativamente reducido (~3× más barato que la tarifa on-demand). Para entrenamientos largos, el equipo debió implementar checkpoint frecuente y reanudación automática para resistir las preemptions. Es una técnica estándar para grupos académicos con presupuesto limitado.

### 8.3 TFRC — TensorFlow Research Cloud

Aunque el paper no menciona TFRC explícitamente en el cuerpo (solo agradece "Google for helping us with the Cloud TPU program for research" en Acknowledgments), el **TensorFlow Research Cloud (TFRC)** era el programa específico de Google que regalaba acceso gratuito a TPUs a grupos académicos seleccionados durante 2019-2020. TFRC fue clave en la democratización del compute para la academia: muchos modelos BERT-monolingüe (CamemBERT, BERTje, BETO, BERTimbau, FinBERT) se entrenaron con TPUs donadas por TFRC.

El programa TFRC fue posteriormente reemplazado por **TRC (TPU Research Cloud)** y finalmente integrado al programa académico general de Google Cloud. Para 2024-2026 ya no es la fuente principal de compute para investigación; la academia migró parcialmente a GPUs (A100, H100) por sponsorship de NVIDIA o por créditos AWS/Azure. Pero el impacto histórico de TFRC es indiscutible — sin esa donación, BETO probablemente no habría existido en 2020 (el costo a tarifa comercial habría sido prohibitivo para un grupo académico latinoamericano).

---

## 9. GLUES: el benchmark de evaluación

GLUES significa **GLUE for Spanish** y es la segunda contribución del paper. Es una compilación de tareas NLP en español, cada una con dataset estandarizado de train/dev/test, métrica clara y referencias a SOTA previo en mBERT.

### 9.1 Tareas incluidas

| Tarea | Dataset | Tipo | Métrica |
|---|---|---|---|
| **XNLI** | XNLI ES + MNLI traducido | Inferencia textual 3-clases | Accuracy |
| **PAWS-X** | PAWS-X ES | Paraphrasing binaria | Accuracy |
| **NER** | CoNLL-2002 ES (Tjong Kim Sang 2002) | Reconocimiento de entidades nombradas | F1 |
| **POS** | Universal Dependencies v1.4 ES | Etiquetado de partes del discurso | Accuracy |
| **MLDoc** | MLDoc ES (Schwenk & Li 2018) | Clasificación de documentos en 4 categorías Reuters | Accuracy |
| **Dependency Parsing** | UD v2.2 ES (AnCora + GSD) | Parsing de dependencias | UAS / LAS |
| **MLQA, XQuAD, TAR** | Traducciones de SQuAD v1.1 | Question Answering extractivo | F1 / Exact Match |

### 9.2 Resultados: BETO vs mBERT

**Tabla 1 del paper — Resultados en tareas no-QA**:

| Modelo | XNLI | PAWS-X | NER | POS | MLDoc |
|---|---|---|---|---|---|
| Best mBERT | 78.50 | 89.00 | 87.38 | 97.10 | 95.70 |
| **es-BERT uncased** | 80.15 | 89.55 | 82.67 | 98.44 | **96.12** ★ |
| **es-BERT cased** | **82.01** | 89.05 | **88.43** | **98.97** ★ | 95.60 |

★ = nuevo state-of-the-art para la tarea.

**Tabla 2 del paper — Resultados en QA** (F1 / Exact Match):

| Modelo | MLQA→MLQA | TAR→XQuAD | TAR→MLQA |
|---|---|---|---|
| Best mBERT | 53.90 / 37.40 | 77.60 / 61.80 | 68.10 / 48.30 |
| es-BERT uncased | 67.85 / 46.03 | 77.52 / 55.46 | 68.04 / 45.00 |
| es-BERT cased | 68.01 / 45.88 | 77.56 / 57.06 | 69.15 / 45.63 |

### 9.3 Análisis de los resultados

**Donde BETO gana decisivamente**:

- **XNLI**: +3.5 accuracy (82.01 vs 78.50). La inferencia textual es una tarea donde el modelo necesita representaciones semánticas profundas, no atajos léxicos. La superioridad del vocab español dedicado y el corpus 6× mayor en español se materializan aquí.
- **POS Tagging**: +1.87 acc (98.97 vs 97.10), nuevo SOTA. La morfología española se beneficia directamente de un vocab dedicado.
- **NER**: cased +1.05 F1, pero uncased pierde -4.71 F1 (82.67 vs 87.38). NER es una tarea sensible a mayúsculas (los nombres propios suelen ir capitalizados), así que la pérdida en uncased es esperable. La versión cased domina mBERT.
- **MLQA→MLQA**: +14.11 F1 (68.01 vs 53.90). El paper sugiere que esto se debe parcialmente a la mala calidad de las traducciones MLQA (~50% de mismatches entre answer y starting position), que afecta tanto a mBERT como a BETO pero BETO al menos tiene mejor cobertura léxica.

**Donde BETO empata o pierde**:

- **PAWS-X**: empate prácticamente (89.55 vs 89.00). La paráfrasis adversarial de PAWS está diseñada explícitamente para ser difícil para modelos basados en bolsa de palabras, y BETO no aporta ventaja arquitectónica para esto.
- **TAR→XQuAD**: empate (~77.5 ambos modelos en F1, pero mBERT mejor en Exact Match 61.8 vs 57.06). El paper sugiere que mBERT puede aprovechar mejor el train inglés original cuando hay translation noise.
- **MLDoc**: BETO uncased gana, cased empata. Las cuatro categorías Reuters (CCAT, ECAT, GCAT, MCAT) son distinguibles por vocabulario común, así que la ventaja léxica de BETO ayuda pero no mucho.

**Crítica**: el paper compara contra "best mBERT result reported in the literature for the same setting". Esto introduce variabilidad: diferentes papers reportan mBERT con diferentes regímenes de fine-tuning. Una comparación ideal habría fine-tuneado mBERT en idéntico setting que BETO. El paper reconoce esta limitación de forma implícita ("We use the hyperparameters recommended by Devlin et al. 2018") pero no la cuantifica.

### 9.4 Comparación contra XLM-R

El paper mismo reconoce (Sección 5.2) que **XLM-RoBERTa (Conneau et al. 2019)** con 560M parámetros y entrenado en CommonCrawl multilingüe alcanza **85.6% en XNLI y 89% en NER** español — superando a BETO en ambas tareas. Esto es importante: BETO no es el SOTA absoluto en español ya en 2020; un modelo multilingüe grande y bien entrenado lo supera. La justificación para usar BETO es:

1. **Menor costo de inferencia** (110M vs 560M params).
2. **Menor footprint de memoria** (BETO cabe en una GPU consumer; XLM-R-large requiere V100 o más).
3. **Fine-tuning más rápido** y estable en datasets pequeños.
4. **Vocabulario específico** que reduce tokens por palabra en español.

BETO está pensado como modelo de trabajo cotidiano (production-grade), no como SOTA leaderboard.

---

## 10. Limitaciones

### 10.1 Corpus principalmente formal/escrito

Wikipedia, ONU, EU, news y subtítulos cubren razonablemente bien el español formal y el oral preparado, pero **no representan**:

- **Redes sociales**: jerga, abreviaturas (`xq`, `tmb`, `wn`), emojis, hashtags, menciones.
- **Mensajería instantánea**: registros muy informales, ortografía no normativa.
- **Foros y comentarios web**: opiniones, sarcasmo, ironía.
- **Lenguaje hablado espontáneo**: muletillas, disfluencias, code-switching.

Para tareas en estos dominios, BETO underperforms. Esto motivó modelos posteriores como **RoBERTuito** (Pérez et al. 2021, UNC Argentina) específicamente entrenado en Twitter español.

### 10.2 Sin diferenciación de variantes regionales

El corpus mezcla español peninsular (Wikipedia, EU, DOGC), español latinoamericano (news, subtítulos, ONU traducidos), y traducciones mecánicas (parte de OpenSubtitles). No hay esfuerzo deliberado de balance regional ni de etiquetado por variante. Consecuencia:

- El modelo **no distingue voseo rioplatense**, **chileno con uso de "weón"**, **mexicano coloquial**, **andino**, etc.
- Términos sinónimos regionales se tratan como independientes sin reconocer su equivalencia funcional (`computadora`/`computador`/`ordenador`, `aguacate`/`palta`, `auto`/`carro`/`coche`).
- Para aplicaciones específicas (e.g., NER médico en Chile vs Argentina vs México), un practitioner tendría que fine-tunear con datos regionales propios.

### 10.3 Sin Tweets ni dominio médico/legal

El corpus carece deliberadamente de:

- Tweets en español (no estaban en OPUS y crawlearlos requería permisos Twitter API).
- Historiales clínicos (privacidad).
- Jurisprudencia LATAM (disponibilidad heterogénea).

Para estas aplicaciones, el approach estándar es **further pre-training** (continue pre-training BETO sobre el corpus de dominio durante decenas de miles de pasos adicionales antes de fine-tunear). Múltiples grupos han hecho esto: **clinical-beto**, **legal-spanish-bert**, etc.

### 10.4 Modelo "viejo" para 2026

A 6 años de su release, BETO compite con alternativas más modernas:

| Alternativa | Año | Por qué consideraria reemplazo |
|---|---|---|
| **MarIA** (PlanTL-GOB-ES, BSC) | 2022 | RoBERTa-base/large entrenado en corpus BNE 570GB (>10× BETO), licencia abierta gobierno español |
| **RoBERTuito** (UNC Argentina) | 2021 | Específico para Twitter español, mejor en social media |
| **BERTIN** (BSC) | 2021 | RoBERTa-base español con perplexity sampling |
| **mBERT actualizado** | 2023+ | Ha mejorado con más datos |
| **XLM-R-large / XL** | 2020-2022 | Multilingüe top-tier, mejor en español que BETO en muchas tareas |
| **LLaMA-2/3 instruct multilingüe** | 2023-2024 | Para generación y NLU complejo |
| **Aya 23 / Aya Expanse** (Cohere) | 2024 | Multilingüe instruction-tuned |
| **Gemma 2/3** (Google) | 2024-2025 | Multilingüe instruct, eficiente en GPU consumer |

BETO sigue siendo competitivo como **embedding/encoder eficiente** para tareas de clasificación, NER y retrieval donde la latencia importa y un modelo de 110M params es preferible a uno de 7B+. Pero ya no es la elección obvia para tareas nuevas de greenfield.

### 10.5 NSP no se cuestiona

El paper mantiene el objetivo NSP (Next Sentence Prediction) original de BERT, a pesar de que en el momento del paper RoBERTa ya había mostrado (mediados de 2019) que NSP es débil o incluso perjudicial. BETO podría haber adoptado el régimen "MLM-only" de RoBERTa y probablemente habría obtenido ligeras mejoras. Es una decisión conservadora que sigue al BERT canónico aun cuando había evidencia disponible para divergir.

### 10.6 Sin análisis de bias ni cobertura de equidad

El paper no incluye análisis de sesgos del modelo. Investigación posterior (incluyendo el ya mencionado "A Primer in BERTology" de Rogers et al. 2020 y trabajos específicos en BETO) mostró que BETO hereda y amplifica sesgos del corpus: género en profesiones (`enfermera`/`médico`), regional (sobre-representación de español peninsular), socioeconómico, racial. Para aplicaciones en salud o justicia esto requiere auditoría y mitigación independiente del modelo.

### 10.7 Workshop paper, no conferencia top-tier

Es importante mencionar el venue: PML4DC es un **workshop** de ICLR 2020, no track principal. Los workshop papers tienen revisión menos exhaustiva, menor presión por novedad metodológica, y formato más corto. Esto se nota en el paper de BETO: solo 10 páginas, una sección de evaluación corta, sin ablations sistemáticas, sin análisis del comportamiento interno del modelo. **El paper es valioso por el modelo y el corpus liberados, no por la profundidad de su análisis científico**. Esta es una distinción importante para los estudiantes: el impacto real de BETO está en los artefactos públicos (modelos en HF, corpus, benchmark), no en las contribuciones metodológicas del paper.

---

## 11. Variantes y evolución posterior del ecosistema español-BERT

### 11.1 Liberaciones posteriores del DCC UChile

El equipo de Cañete continuó produciendo modelos:

| Modelo | Año aprox. | Descripción |
|---|---|---|
| `dccuchile/distilbert-base-spanish-uncased` | 2020 | DistilBERT español, 6 capas, 66M params, fine-tuning más rápido |
| `dccuchile/albert-base-spanish` | 2020-2021 | ALBERT español (parameter sharing), tamaños base/large/xlarge/xxlarge |
| `dccuchile/roberta-base-spanish` | 2021 | RoBERTa español de DCC UChile |
| Modelos NER, POS, sentiment específicos | 2020-2022 | Versiones fine-tuneadas para tareas específicas, publicadas en HF |

El conjunto `dccuchile/*` en HuggingFace funciona como el **canon español del DCC UChile**. Muchas pipelines empresariales en LATAM cargan de esta organización por defecto.

### 11.2 MarIA (PlanTL-GOB-ES, 2022)

**MarIA** es una familia de modelos publicados por el Plan de Tecnologías del Lenguaje del Gobierno de España, ejecutado por el Barcelona Supercomputing Center (BSC). Incluye:

- `PlanTL-GOB-ES/roberta-base-bne` (~125M params)
- `PlanTL-GOB-ES/roberta-large-bne` (~355M params)
- Versiones GPT (`gpt2-base-bne`, `gpt2-large-bne`)

Entrenado en el **corpus BNE** (Biblioteca Nacional de España) de **570 GB de texto crawleado durante 2009-2019**, deduplicado y filtrado con calidad heurística. Es aproximadamente **15× más datos que BETO**. Como consecuencia:

- MarIA roberta-base supera a BETO en la mayoría de benchmarks GLUES.
- MarIA roberta-large es claramente superior cuando el dominio se acerca al peninsular formal.
- BETO sigue siendo competitivo (a veces superior) cuando el dominio es **LATAM**, **oral**, o **mezclado peninsular/LATAM** — porque el corpus de BETO incluye OpenSubtitles latinoamericano, ONU y otros materiales no-peninsulares que MarIA no ve.

La elección "MarIA vs BETO" depende del dominio downstream.

### 11.3 RoBERTuito (UNC Argentina, 2021)

**RoBERTuito** (Pérez et al. — Juan Manuel Pérez de UNC, no Jorge Pérez del DCC UChile; coincidencia de apellido) es un RoBERTa entrenado específicamente en **500 millones de tweets en español** recolectados con Twitter API. Está diseñado para tareas de social media: análisis de sentimiento, detección de hate speech, ironía, clasificación de tweets. Para esos dominios, RoBERTuito domina a BETO y a mBERT por márgenes amplios.

### 11.4 BERTIN (BSC + comunidad, 2021)

**BERTIN** es un RoBERTa-base español entrenado con un régimen experimental de **perplexity-based sampling**: el corpus Common Crawl español se filtra por perplejidad medida con un modelo de lenguaje pequeño, manteniendo solo textos de calidad intermedia (no demasiado fáciles ni demasiado ruidosos). El resultado es competitivo con MarIA usando menos compute.

### 11.5 Multilingüe big-data: XLM-R, mT5, NLLB, Aya

En paralelo al ecosistema español-monolingüe, los modelos multilingües siguieron creciendo:

- **XLM-R** (2019-2020): RoBERTa entrenado en 2.5TB CommonCrawl 100 idiomas. Variantes base (270M), large (550M), XL (3.5B), XXL (10.7B).
- **mT5** (2020): T5 multilingüe en 101 idiomas.
- **NLLB** (Meta, 2022): No Language Left Behind, traducción 200 idiomas.
- **Aya 23 / Aya Expanse** (Cohere, 2024): instruction-tuned multilingüe, fuerte en español.
- **LLaMA-3 multilingüe**, **Gemma 2 multilingüe**, **Qwen 2.5** (2024): generativos multilingües potentes.

Para 2026, la frontera técnica en NLP español ya no es BETO sino una combinación de:

1. Encoders pequeños y eficientes (BETO, MarIA-base, distilados) para clasificación y retrieval con baja latencia.
2. LLMs multilingües (Aya, LLaMA, Qwen, Gemini, GPT-4o) para generación, NLU compleja y few-shot reasoning.

BETO ocupa el primer nicho con dignidad.

---

## 12. Conexión con la clase 20 del Diplomado IA UC

La clase 20 explora la **transición de embeddings estáticos a representaciones contextualizadas**, con BERT como caso central. BETO funciona en la clase como **caso canónico de transferencia de arquitecturas anglo a otros idiomas**:

- **No reinventa la arquitectura**. Reusa BERT-base exactamente. Esto enseña que el aporte de una contribución no siempre es metodológico — a veces es el **ensamble correcto de corpus + tokenizador + régimen de entrenamiento + benchmark de evaluación + liberación pública**.
- **Demuestra el valor del vocabulario dedicado**. Las ganancias sobre mBERT en XNLI, POS y NER son atribuibles principalmente al WordPiece español de 31K vs el WordPiece multilingüe de 119K dividido entre 104 idiomas.
- **Ilustra el patrón "arquitectura oficial + pesos comunitarios"**. La carga `BertModel.from_pretrained("dccuchile/...")` es el caso de uso paradigmático en HuggingFace.
- **Conecta con la realidad LATAM del estudiante del Diplomado**. La clase 20 no es solo sobre BERT en inglés — es sobre cómo aplicar estas ideas en el contexto local. BETO es la herramienta concreta que el practitioner chileno o LATAM usa antes de considerar alternativas.

La clase 20 cubre además:

- **ELMo** (Peters et al. 2018) — contraste con BETO en bidireccionalidad shallow vs deep.
- **GPT family** (Radford et al. 2018, 2019, 2020) — contraste con BETO en unidireccional vs bidireccional, generación vs representación.
- **ChatGPT y RLHF** — contraste paradigmático con la era pre-instructiva donde BETO operaba.

BETO se sitúa en el **inicio** del paradigma "pre-train un encoder bidireccional, fine-tune para tu tarea". Es el último momento donde esa receta era el approach de elección. Para 2026, el approach dominante es "tomar un LLM grande y few-shot prompt", y BETO sobrevive en los nichos donde la eficiencia importa más que la flexibilidad.

---

## 13. Conexión con el Lab 20

El Lab 20 incluye una sección práctica con BETO en las **celdas 25-33** del notebook. Los estudiantes:

### 13.1 Cargan el modelo y el tokenizador

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
```

Nota importante: **se carga con `BertModel`, no con "BetoModel"**. La clase `BetoModel` no existe en HuggingFace. Este punto se discute en el lab como ejemplo del patrón "arquitectura oficial + pesos comunitarios". Es identical al patrón de `RoBertModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")` (MarIA) o `BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")` (BERTimbau).

### 13.2 Tokenizan una oración española

```python
inputs = tokenizer("Hola Mundo!", return_tensors="pt")
print(inputs)
# {'input_ids': tensor([[CLS_id, hola_id, mundo_id, !_id, SEP_id]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1]])}
```

El output tiene los mismos tres tensores que en BERT inglés:

- `input_ids`: IDs de los subwords incluyendo `[CLS]` al inicio y `[SEP]` al final.
- `token_type_ids`: vector de segment IDs (0 para single-sentence, 0/1 para sentence-pair).
- `attention_mask`: vector binario 1=token real, 0=padding.

### 13.3 Comparación token-a-token con BERT inglés

Una actividad útil del lab es comparar cómo tokeniza BETO vs `bert-base-uncased` la misma idea:

| Texto | Tokenizador | Tokens producidos |
|---|---|---|
| "tokenization" | BERT inglés uncased | `['token', '##ization']` |
| "tokenización" | BERT inglés uncased | `['token', '##iza', '##cion']` (sin acentos) |
| "tokenización" | BETO uncased | `['tokenizacion']` o `['token', '##izacion']` |
| "tokenización" | BETO cased | `['tokenización']` |
| "comerían" | BETO cased | `['comerían']` o `['com', '##erían']` |
| "comerían" | mBERT uncased | `['co', '##mer', '##ían']` típico |

La fragmentación menor en BETO es directamente observable. Esto es la ganancia material del vocab dedicado.

### 13.4 Forward pass y representaciones

```python
import torch
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: shape (1, seq_len, 768)
# outputs.pooler_output: shape (1, 768) — representación del [CLS] proyectada
```

Las representaciones obtenidas son contextuales: el mismo token tiene embeddings distintos según la oración. Los estudiantes pueden inspeccionar similitud coseno entre embeddings de la misma palabra en contextos distintos (`"banco" del río` vs `"banco" comercial`) para verificar empíricamente que BETO captura sentido contextual — algo que los embeddings estáticos del Lab 18 (Word2Vec, GloVe español) no podían hacer.

### 13.5 Pipeline de uso típico downstream

Para tareas reales en español:

```python
from transformers import pipeline

# Análisis de sentimiento (usar checkpoint fine-tuneado)
sentiment = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")
print(sentiment("Me encanta este producto, es maravilloso"))

# NER
ner = pipeline("ner", model="dccuchile/bert-base-spanish-wwm-cased",
               tokenizer="dccuchile/bert-base-spanish-wwm-cased")
print(ner("José Cañete trabajó en la Universidad de Chile."))

# Question answering
qa = pipeline("question-answering",
              model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
print(qa(question="¿Quién creó BETO?", context="BETO fue creado por José Cañete..."))
```

Los modelos fine-tuneados sobre BETO para tareas específicas son numerosos en HF — la organización `dccuchile/` solo tiene los modelos base; la comunidad ha publicado cientos de versiones fine-tuneadas para tareas particulares.

### 13.6 El patrón "arquitectura oficial + pesos comunitarios"

Este es el insight conceptual transversal de la celda BETO en el lab. La idea:

- **HuggingFace mantiene un conjunto finito de arquitecturas oficiales** (`BertModel`, `RobertaModel`, `GPT2Model`, `T5Model`, `LLamaModel`, etc.).
- **Cualquiera puede entrenar pesos y publicarlos** bajo una organización HF, especificando en `config.json` qué arquitectura oficial deben usar.
- **`from_pretrained()` carga los pesos en la arquitectura especificada**, sin requerir código custom.
- Este patrón hace que **un modelo nuevo (BETO, MarIA, RoBERTuito) no requiera librería propia**; basta el repo de pesos.

Para el estudiante, este patrón generaliza: cuando aparezca el próximo modelo (hipotético `dccuchile/llama-3-spanish-chile`), no necesita aprender librería nueva — `LlamaForCausalLM.from_pretrained("dccuchile/llama-3-spanish-chile")` lo carga.

---

## 14. Aplicaciones reales en LATAM

BETO se ha usado masivamente en aplicaciones de producción en LATAM. Algunos casos documentados o conocidos:

### 14.1 NER médico

Hospitales y aseguradoras en Chile, Argentina, México y Colombia han fine-tuneado BETO para:

- **Extracción de medicamentos, dosis y vías** en historiales clínicos para alimentar bases CIE-10 / ATC.
- **Identificación de diagnósticos y comorbilidades** desde texto libre.
- **De-identificación automática** (NER de nombres, RUTs, direcciones, fechas) para cumplir leyes de protección de datos (Ley 21.719 chilena, Ley 25.326 argentina).
- **Codificación SNOMED CT en español** mediante BETO + diccionarios + reglas.

Modelos disponibles: `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, `mrm8488/bert-spanish-cased-finetuned-ner`, varios checkpoints de `dccuchile/`.

### 14.2 FHIR record matching en español

El matching probabilístico de historiales médicos (problema MDM — Master Data Management) en hospitales LATAM requiere comparar nombres, direcciones y datos demográficos escritos con variaciones ortográficas, abreviaturas, errores de digitación. BETO se usa como **encoder en bi-encoders** que producen embeddings de registros para clustering y blocking de candidatos, antes de pasar al scorer (regresión logística, GBM, o transformer cross-encoder) que decide el match final. Para registros en español, BETO supera a embeddings monolingües construidos sobre fastText o GloVe.

### 14.3 Análisis de sentimiento

Empresas de retail, banca y telcos en LATAM clasifican opiniones de clientes (encuestas NPS, reviews, redes sociales) con BETO fine-tuneado en datasets como TASS (corpus de análisis de sentimiento español) o datos internos. Modelos populares: `finiteautomata/beto-sentiment-analysis`, `pysentimiento/robertuito-sentiment-analysis` (para Twitter), `nlptown/bert-base-multilingual-uncased-sentiment` (multilingüe baseline).

### 14.4 Clasificación legal y regulatoria

Estudios jurídicos y bancos clasifican documentos contractuales, regulaciones y jurisprudencia con BETO fine-tuneado. Hay datasets públicos como **Spanish Legal Text Corpus** y varios trabajos del IMFD que aplican BETO específicamente a corpus jurídicos chilenos (Código Civil chileno, jurisprudencia de la Corte Suprema).

### 14.5 Sistemas conversacionales

Adereso (la startup mencionada en el paper) y otras empresas LATAM (Botmaker, Cliengo, Yalo) usan BETO como NLU backbone en chatbots empresariales: clasificación de intenciones, extracción de entidades de usuario, scoring de similitud entre query y FAQ. Para sistemas en español, BETO sigue siendo competitivo con LLMs por consideraciones de latencia y costo.

### 14.6 Educación

Diplomados y cursos de IA en Chile (UC, UChile, Universidad de los Andes, UTFSM) usan BETO como modelo de demostración en sus módulos de NLP. Es el ejemplo natural cuando se enseña "BERT" porque está en español, es chileno, y los estudiantes pueden cargarlo y probarlo en minutos.

---

## 15. Notas para integrar al site

Cosas que el `papers/beto-canete-2020.md` actual (si existe) debería cubrir y que pueden tomarse de este documento sin duplicar contenido:

1. **Discrepancia arquitectura paper vs config.json**: el paper dice 16 heads / hidden 1024, pero el modelo real en HF es 12 heads / hidden 768 (BERT-base estándar).
2. **Vocabulario SentencePiece BPE 31K + 1K placeholders = 32K tokens**, pero expuesto al usuario como WordPiece (`##` continuación) vía `BertTokenizer`.
3. **Dynamic Masking 10x** (no la dinámica pura de RoBERTa, sino 10 patrones pre-computados por oración).
4. **Whole-Word Masking** y por qué impacta más en español que en inglés.
5. **Schedule de dos fases** (batch 2048 seq 128 → batch 256 seq 512) inspirado en You et al. 2019.
6. **TPU v3-8 preemptible vía TFRC** y el rol histórico de TFRC en democratizar compute académico.
7. **Tabla GLUES con números literales** (Tablas 1 y 2 del paper).
8. **Mención explícita a XLM-R superando a BETO en XNLI/NER** según el propio paper.
9. **Versiones del ecosistema dccuchile**: ALBERT, DistilBETO, RoBERTa-base-spanish.
10. **Alternativas modernas a considerar para greenfield 2026**: MarIA, RoBERTuito, BERTIN, XLM-R, LLMs multilingües.
11. **Patrón "arquitectura oficial + pesos comunitarios"** como insight transversal para HuggingFace.

El `fundamentos/beto-spanish-bert.md` (si existe en el site) puede enfocarse en uso práctico, dejando este documento como referencia técnica más profunda.

---

## 16. Lectura recomendada complementaria

- **Wu & Dredze (2019)** — *Beto, Bentz, Becas: The surprising cross-lingual effectiveness of BERT*. Paper que da el nombre coincidente al modelo chileno (ironía histórica) y analiza mBERT en múltiples idiomas. Referencia obligada para cualquier discusión BETO vs mBERT.
- **Pires, Schlinger & Garrette (2019)** — *How multilingual is multilingual BERT?* Análisis cuantitativo de las capacidades cross-lingüe de mBERT, motivación implícita para modelos monolingües como BETO.
- **Martin et al. (2019) — CamemBERT** y **Le et al. (2019) — FlauBERT** — los dos BERTs franceses contemporáneos a BETO, con análisis más extenso que conviene comparar.
- **Conneau et al. (2019) — XLM-R** — el modelo multilingüe que BETO mismo reconoce como superior en algunas tareas. Lectura para entender el approach multilingüe scaled.
- **Liu et al. (2019) — RoBERTa** — base técnica de varias decisiones de entrenamiento de BETO (Dynamic Masking, batches grandes, más pasos).
- **Cui et al. (2019) — Pre-training with Whole Word Masking for Chinese BERT** — paper original de WWM, aplicado primero a chino, adoptado por BETO.
- **You et al. (2019) — LAMB optimizer** — justificación del schedule de dos fases con batch grande.
- **Kudo & Richardson (2018) — SentencePiece** — tokenizador usado para construir el vocabulario de BETO.
- **MarIA papers (Gutiérrez-Fandiño et al. 2022)** — alternativa española posterior a BETO con corpus mucho mayor.
- **RoBERTuito (Pérez et al. 2021)** — alternativa española específica para redes sociales.
- **Pyysalo, Kanerva et al. — Spanish biomedical corpora** — para entender el ecosistema biomédico en español que extiende BETO.
- **A Primer in BERTology (Rogers et al. 2020)** — survey general de análisis interno de BERT, aplicable conceptualmente a BETO.
