---
title: "Entrenamiento y Evaluación"
weight: 30
math: true
---

## Función de pérdida

Las 18 categorías son mutuamente excluyentes — el problema es **clasificación multiclase**:

```python
criterion = nn.CrossEntropyLoss()
```

`CrossEntropyLoss` recibe el output crudo (logits, sin softmax) y el índice de la clase verdadera. Internamente aplica `log_softmax` + `nll_loss`. Por eso los modelos no aplican softmax al final — entregan logits directamente.

---

## SGD hecho a mano

Este lab no usa `torch.optim.SGD`. La actualización se hace explícitamente recorriendo los parámetros:

```python
learning_rate = 0.005

def train(category_tensor, line_tensor, model):
    hidden = model.init_hidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):                  # recorre carácter a carácter
        if model.bidirectional:
            output, hidden = model(
                (line_tensor[i], line_tensor[line_tensor.size()[0]-i-1]),
                hidden,
            )
        else:
            output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)               # solo el último output cuenta
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)      # SGD manual

    return output, loss.item()
```

Tres detalles importantes:

1. **Solo el último `output` aporta al loss.** La red lee los N caracteres del apellido y al final emite una predicción. Los outputs intermedios se descartan (este es un patrón **many-to-one**).
2. **`loss.backward()` recorre toda la secuencia.** Como `hidden` arrastra el grafo de cómputo desde el primer carácter, el backward es **Backpropagation Through Time (BPTT)** completo sobre el largo del apellido.
3. **batch=1.** Cada iteración procesa un solo apellido. Sin padding ni batching real (eso requeriría `pack_padded_sequence`).

`learning_rate = 0.005` es bajo precisamente porque BPTT es propenso a explosión de gradientes — un lr más alto haría diverger especialmente a la RNN vanilla.

> **Nota — comparación con el tutorial padre:** el [tutorial oficial de PyTorch](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) en el que se basa este lab usa `lr=0.15` (30× más alto) **pero con gradient clipping** a norma 3.0. Sin clipping, ese lr divergería instantáneamente. El lab UC opta por el camino opuesto: sin clipping, con un lr conservador de 0.005. Ambas estrategias son válidas — la de PyTorch entrena más rápido, la del lab es más simple de implementar y razonar.

---

## Loop de entrenamiento

```python
def train_model(model, n_iters=200000, print_every=10000, plot_every=1000):
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, model)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' %
                  (iter, iter/n_iters*100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses, model
```

200 000 iteraciones × batch 1 ≈ equivalente a unas 10 épocas sobre todo el dataset (hay ~20K apellidos en total). El loss promedio se almacena cada 1 000 iteraciones para suavizar el gráfico.

---

## Evaluación: matriz de confusión

Para diagnosticar qué clases confunde el modelo, se construye una matriz `(18, 18)` normalizada por filas:

```python
def compute_confusion_matrix(model):
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor, model)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()    # normaliza por fila

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
```

La fila `i` muestra, sobre todos los apellidos reales del idioma `i`, cómo se distribuyeron las predicciones. La diagonal son los aciertos. **Confusiones típicas esperadas:** Korean ↔ Vietnamese, Spanish ↔ Italian/Portuguese, English ↔ Scottish — idiomas con sistemas onomásticos parecidos romanizados al mismo alfabeto.

---

## Resultados de las 3 arquitecturas

Tras entrenar las tres baselines durante 200,000 iteraciones cada una (batch=1, lr=0.005, sin gradient clipping), estos son los resultados.

### RNN vanilla — `n_hidden = 147`

**Tiempo de entrenamiento:** 5m 7s (el más rápido de los tres).

```text
10000   5%  (0m 15s)  2.8276  Kuiper / German       ✗ (Dutch)
30000  15%  (0m 46s)  1.7636  Krauss / Dutch        ✗ (German)
50000  25%  (1m 16s)  1.5948  Ritchie / French      ✗ (Scottish)
70000  35%  (1m 47s)  0.1502  Ta / Vietnamese       ✓
90000  45%  (2m 17s)  1.1587  Venne / Dutch         ✓
130000 65%  (3m 19s)  2.2403  Atkin / Russian       ✗ (English)
170000 85%  (4m 21s)  0.1142  Baz / Arabic          ✓
200000 100% (5m 7s)   1.0785  Bosch / Czech         ✗ (German)
```

![Curva de pérdida RNN vanilla](/laboratorios/lab-11/loss-rnn-vanilla.png)

La curva arranca en pérdida ~2.85 (cercana a la pérdida aleatoria $\ln 18 \approx 2.89$) y cae rápido en los primeros 25K iters hasta ~1.5. Desde ahí el descenso es gradual hasta **~0.65–0.70 al final**.

![Matriz de confusión RNN vanilla](/laboratorios/lab-11/cm-rnn-vanilla.png)

Diagonal fuerte (precisión ≳ 0.8) en idiomas con grafías muy distintivas tras la romanización: Korean, Greek, Polish, Vietnamese, Japanese, Arabic. Diagonal media (~0.5–0.7) en Chinese, Scottish, Italian, Czech, Portuguese, Irish, French, Russian, Dutch, German. Las **confusiones esperadas se cumplen**: English se dispersa hacia Scottish e Irish; Spanish hacia Italian y Portuguese; German y Dutch se confunden mutuamente.

---

### LSTM — `n_hidden = 64`

**Tiempo de entrenamiento:** 7m 47s (~50% más lento que la RNN vanilla por las cuatro compuertas).

```text
10000   5%  (0m 23s)  2.9363  Kau / French          ✗ (Chinese)
30000  15%  (1m 10s)  1.7540  Ma / Vietnamese       ✗ (Korean)
60000  30%  (2m 19s)  0.6354  Ruslanov / Russian    ✓
80000  40%  (3m 7s)   0.3023  Gallchobhar / Irish   ✓
100000 50%  (3m 53s)  0.9934  Guerra / Portuguese   ✓
130000 65%  (5m 3s)   0.8223  Moreno / Portuguese   ✓
170000 85%  (6m 36s)  0.0919  Chikanatsu / Japanese ✓
200000 100% (7m 47s)  0.3592  Oh / Korean           ✓
```

![Curva de pérdida LSTM](/laboratorios/lab-11/loss-lstm.png)

Plateau plano los primeros ~20K iters (la red tarda más en arrancar — más parámetros que reordenar para sincronizar las cuatro compuertas). Caída pronunciada entre iter 20K y 50K, luego descenso gradual hasta **~0.85–0.95 al final**.

![Matriz de confusión LSTM](/laboratorios/lab-11/cm-lstm.png)

Patrón muy similar al RNN vanilla: misma diagonal fuerte en idiomas distintivos y misma debilidad en familias romances y germánicas. **English se nota algo más confundido** que en el RNN vanilla, dispersándose a Czech y Russian.

---

### BiLSTM — `n_hidden = 40`

**Tiempo de entrenamiento:** 13m 37s (el más lento — procesa cada carácter dos veces, forward y backward).

```text
10000   5%  (0m 40s)  2.8077  Costantini / Portuguese  ✗ (Italian)
30000  15%  (2m 2s)   2.5924  Carey / German           ✗ (Irish)
50000  25%  (3m 23s)  1.0477  Acquati / Italian        ✓
80000  40%  (5m 26s)  3.8204  Dimmock / Polish         ✗ (English)
100000 50%  (6m 47s)  0.1499  Fujimaki / Japanese      ✓
130000 65%  (8m 48s)  0.4356  Huan / Chinese           ✓
170000 85%  (11m 33s) 0.0891  Raghailligh / Irish      ✓
200000 100% (13m 37s) 0.0261  Kitoaji / Japanese       ✓
```

![Curva de pérdida BiLSTM](/laboratorios/lab-11/loss-bilstm.png)

**Plateau más largo de los tres** (~25K iters totalmente plano) — al ser dos LSTMs con `n_hidden=40` cada una, hay más parámetros que sincronizar antes de que aparezca señal. Después, descenso continuo hasta **~0.95–1.05 al final**.

![Matriz de confusión BiLSTM](/laboratorios/lab-11/cm-bilstm.png)

Diagonal comparable al LSTM no bidireccional. La bidireccionalidad **no rinde una mejora visible** en este lab — los apellidos son cortos y la información discriminativa está distribuida a lo largo de la secuencia, no concentrada al final donde un LSTM unidireccional la perdería.

---

### Comparativa de las 3 baselines

| Modelo | $n\_hidden$ | Parámetros | Tiempo | Loss final (suavizada) |
|--------|------------:|-----------:|-------:|:---:|
| RNN vanilla (Elman) | 147 | 32,946 | **5m 7s** | **~0.65–0.70** ← menor |
| LSTM | 64 | 32,658 | 7m 47s | ~0.85–0.95 |
| BiLSTM | 40 | 33,138 | 13m 37s | ~0.95–1.05 |

**A paridad de parámetros (~33K), el RNN vanilla gana.** Es contraintuitivo si uno espera "LSTM > RNN siempre", pero tiene sentido en este régimen:

1. **Apellidos cortos (≲ 12 caracteres):** el vanishing gradient — el problema que LSTM resuelve — no muerde a esta longitud. La red solo necesita "recordar" 5–10 pasos hacia atrás, algo que una RNN vanilla maneja sin problema.
2. **Igualar parámetros castiga a las arquitecturas con compuertas:** la LSTM/BiLSTM se quedan con `n_hidden` chico (64 / 40) — es decir, con menos capacidad representacional **por step**. La RNN aprovecha sus 147 dimensiones de hidden state directamente.
3. **Las matrices de confusión son casi idénticas:** los tres modelos cometen los mismos errores (English↔Scottish, Spanish↔Italian, German↔Dutch). Eso indica que el cuello de botella no es la arquitectura recurrente, sino la **información disponible** — el alfabeto romanizado borra distinciones onomásticas reales.

**Implicación pedagógica:** las gates y la bidireccionalidad son herramientas con un costo (más parámetros, entrenamiento más lento, más tiempo para arrancar). Solo pagan el costo cuando el problema lo demanda — secuencias largas, dependencias distantes, contexto futuro. Para clasificación de tokens cortos, la RNN vanilla bien dimensionada es competitiva.

---

## Predicción manual

Para probar el modelo entrenado con apellidos arbitrarios:

```python
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line), model)
        topv, topi = output.topk(n_predictions, 1, True)
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))

predict('Dovesky')   # esperado: Russian / Czech / Polish
predict('Jackson')   # esperado: Scottish / English
predict('Satoshi')   # esperado: Japanese
predict("Muñoz")     # esperado: Spanish (la ñ se convierte a 'n' por unicodeToAscii)
predict("Raymond")   # esperado: French / English
```

Los valores son **logits crudos** (no probabilidades) — sirven para ranking pero no son interpretables como porcentajes.

### Resultados sobre el BiLSTM entrenado

```text
> Dovesky
(6.48) Czech
(6.25) Russian
(5.23) Polish

> Jackson
(5.94) Scottish
(4.68) English
(3.09) German

> Satoshi
(7.72) Japanese
(5.68) Arabic
(2.94) Italian

> Muñoz
(4.89) Spanish
(3.88) Portuguese
(2.18) Polish

> Raymond
(4.12) French
(3.83) English
(3.16) Scottish
```

**Las 5 predicciones top-1 son correctas** y los top-3 capturan los idiomas esperados:

- **Dovesky** → Czech (6.48), seguido por Russian (6.25) y Polish (5.23). Los tres slavos esperados están en el top 3, con Czech apenas por delante de Russian — el sufijo `-sky` es ambiguo entre eslavos occidentales y rusos.
- **Jackson** → Scottish (5.94) > English (4.68). Los dos esperados son top 2, con margen claro.
- **Satoshi** → Japanese (7.72) con margen amplio sobre Arabic (5.68). El sufijo `-shi` es muy distintivo del japonés.
- **Muñoz** → Spanish (4.89). Aunque la `ñ` se pierde (se convierte en `n` por `unicodeToAscii`), el patrón general del apellido sigue siendo claramente hispano.
- **Raymond** → French (4.12) > English (3.83) > Scottish (3.16). Margen pequeño entre French e English — coherente con la confusión Western European visible en la matriz de confusión.
