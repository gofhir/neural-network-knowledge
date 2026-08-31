---
title: "Fake news y el atajo de Reuters"
weight: 4
math: true
---

Es el único entrenamiento real del laboratorio: fine-tuning de `bert-base-uncased` para clasificar noticias como reales o falsas. Funciona —la accuracy sube rápido— y por eso conviene preguntarse **qué aprendió exactamente**.

## El dataset

El CSV proviene del *Real and Fake News Dataset* de Kaggle. Su construcción es la parte que hay que mirar antes que el código: las noticias **reales** vienen mayoritariamente de **Reuters**, y las **falsas** de una colección heterogénea de sitios.

{{< concept-alert type="clave" >}}
**Reuters tiene un manual de estilo.** Sus despachos empiezan con una línea de fecha y lugar (`WASHINGTON (Reuters) -`), usan verbos de atribución de forma sistemática, evitan adjetivos valorativos y siguen una estructura de pirámide invertida rigurosa.

Un clasificador que aprende a reconocer ese registro alcanza accuracy alta **sin evaluar factualidad alguna**. No está distinguiendo verdadero de falso: está distinguiendo *escrito por una agencia de noticias* de *escrito por otra cosa*.

La prueba de que el atajo existe es lo que pasa fuera del dataset: ese mismo modelo clasificaría como "real" una noticia inventada redactada en estilo Reuters, y como "falsa" un reportaje verídico de un blog. **La accuracy alta y la inutilidad práctica conviven sin contradicción.**
{{< /concept-alert >}}

Es un caso de libro de *shortcut learning*, y tiene un pariente cercano en el mismo laboratorio: [SWAG](/papers/swag-zellers-2018) fue construido con *Adversarial Filtering* precisamente para cerrar atajos de este tipo, generando distractores que resisten a los modelos estilísticos. Aquí la puerta quedó abierta de par en par.

La forma de verificarlo, si se quisiera: eliminar la línea de fecha y las menciones a `(Reuters)` del texto, reentrenar y comparar. Si la accuracy se desploma, el atajo era el modelo.

## Ocho defectos del pipeline

El código de entrenamiento es un tutorial adaptado, y arrastra problemas que vale la pena separar por gravedad.

### 1 · La truncación que no coincide

```python
# celda 39 — preprocesamiento
df_raw['text'] = df_raw['text'].apply(lambda x: ' '.join(x.split(maxsplit=200)[:200]))
```
```python
# celda 42 — tokenización
MAX_SEQ_LEN = 128
```

El preprocesamiento conserva **200 palabras**; el tokenizador corta en **128 tokens**. Como WordPiece fragmenta, 128 tokens son aproximadamente **80–95 palabras** de texto real.

$$200 \text{ palabras preparadas} \quad \longrightarrow \quad \sim 90 \text{ palabras vistas por el modelo}$$

**Más de la mitad del texto preparado se descarta.** No es un error que rompa nada — es trabajo de preprocesamiento tirado a la basura, y una discrepancia que induce a creer que el modelo ve más contexto del que ve.

### 2 · El `attention_mask` descartado

```python
tokenized_text = self.tokenizer(titletext,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len,
                                return_tensors='pt')['input_ids']    # <- solo input_ids
return label, tokenized_text
```

El tokenizador devuelve `input_ids` **y** `attention_mask`. El `Dataset` se queda solo con el primero.

Consecuencia: como se usa `padding='max_length'`, **todas** las secuencias llegan a 128 tokens rellenando con `[PAD]`, y sin máscara **el modelo atiende sobre el padding como si fuera contenido**. Una noticia de 30 tokens reales aporta 98 posiciones de relleno que participan en cada capa de auto-atención.

Es el defecto más serio de la lista: degrada de verdad, y en silencio.

### 3 · Acceso por posición de columna

```python
self.labels     = self.df.iloc[:, 0]
self.titletexts = self.df.iloc[:, 3]
```

Funciona porque la celda 39 hace `reindex(columns=['label','title','text','titletext'])` justo antes. Basta con reordenar esa lista, agregar una columna o releer un CSV con otro orden para que el modelo entrene **etiquetas contra el texto equivocado** sin ningún aviso. `self.df['label']` cuesta lo mismo y no tiene ese modo de falla.

### 4 · Una loss declarada que nunca se usa

```python
def train(model, optimizer, criterion = nn.BCELoss(), ...):
```

`BertForSequenceClassification` **calcula su propia loss internamente** cuando se le pasan `labels`: con `num_labels=2` usa `CrossEntropyLoss`. El `criterion` del argumento por defecto nunca se invoca.

Además, `nn.BCELoss` sería incorrecto aquí por partida doble: espera probabilidades en $[0,1]$ (no logits) y está pensada para una salida escalar, no para dos clases. Código muerto que sugiere una arquitectura distinta de la que efectivamente corre.

### 5 · Validación no determinista

```python
dev_loader = DataLoader(dev_dataset, batch_size=16, num_workers=1, shuffle=True)
```

Barajar en validación no cambia la métrica agregada, pero **hace irreproducibles las curvas de loss por paso** cuando se evalúa sobre lotes parciales. `test_loader` sí usa `shuffle=False`, así que es una inconsistencia dentro de la misma celda.

### 6 · `Adam` en lugar de `AdamW`

```python
optimizer = optim.Adam(model.parameters(), lr=2e-5)
```

El paper de BERT usa **AdamW**, con *weight decay desacoplado*. La diferencia no es cosmética: en Adam el término de decaimiento entra dentro del gradiente adaptativo y termina escalado por el segundo momento, lo que lo vuelve inconsistente entre parámetros. Con `weight_decay=0` —el default— ambos coinciden, así que aquí no cambia nada; pero la línea invita a agregar decay y ahí sí empezaría a divergir del régimen del paper.

### 7 · Sin scheduler, sin clipping, sin precisión mixta

El fine-tuning de BERT canónico usa **warmup lineal seguido de decaimiento** y *gradient clipping*. Aquí el learning rate es constante en $2\times10^{-5}$ y no hay clipping. Funciona porque una época sobre un dataset pequeño no da tiempo a divergir, pero es un régimen distinto del que reportan los papers.

### 8 · Una sola época

```python
train(model=model, optimizer=optimizer, num_epochs=1)
```

La firma declara `num_epochs=5` y la llamada pasa 1. Subentrenado, pero suficiente para el propósito pedagógico — y honesto en el sentido de que no pretende otra cosa.

## Un detalle que funciona por accidente

```python
labels = labels.type(torch.LongTensor).unsqueeze(1)     # shape (B, 1)
```

`CrossEntropyLoss` espera etiquetas de forma $(B,)$, no $(B, 1)$. No falla porque `BertForSequenceClassification` aplica internamente `labels.view(-1)`, que aplana. El `unsqueeze(1)` es innecesario y se salva por una línea de la librería.

## Qué queda de todo esto

El clasificador **funciona**: entrena, converge y clasifica bien el conjunto de prueba. Los ocho puntos anteriores no impiden eso.

Lo que sí conviene llevarse es el orden de importancia. De los ocho defectos, el que más degrada es el `attention_mask` descartado; el que más riesgo latente tiene es el acceso por posición de columna. Pero **ninguno de los ocho es el problema principal del experimento**.

El problema principal es el dataset: un modelo perfecto entrenado sobre estos datos aprendería a reconocer el estilo de Reuters. Arreglar los ocho defectos daría un clasificador de estilo periodístico mejor entrenado — no un detector de desinformación.

---

**Siguiente:** [GPT-2 y los límites del contexto](05-gpt-2-y-los-limites-del-contexto) — la última sección del laboratorio.
