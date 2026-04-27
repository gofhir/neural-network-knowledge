---
title: "Resolución del Laboratorio"
weight: 50
math: true
---

Resolución completa de las 2 actividades prácticas del laboratorio.

{{< notebook-viewer src="/notebooks-html/lab11.html" >}}

---

## Actividad 1 — Cantidad de parámetros de los 3 modelos

### Enunciado

Mencione la dimensionalidad oculta de los 3 modelos, calcule su cantidad de parámetros y explique por qué **no sería justo** usar dimensionalidad oculta pareja para todos los modelos.

### Dimensionalidades del lab

| Modelo | $n\_hidden$ |
|--------|-------------|
| RNN vanilla (Elman) | **147** |
| LSTM | **64** |
| BiLSTM | **40** |

### Conteo de parámetros

Las constantes del problema son $n\_letters = 57$ (vocabulario) y $n\_categories = 18$ (idiomas).

Cada `nn.Linear(in, out)` aporta `in·out + out` parámetros (peso + bias). Cada `nn.LSTMCell(in, h)` aporta `4·h·in + 4·h·h + 4·h + 4·h` (cuatro compuertas con dos sesgos en la API de PyTorch).

#### RNN vanilla, $n\_h = 147$

| Capa | Cálculo | Parámetros |
|------|---------|-----------:|
| `x2h = Linear(57, 147)` | $57 \cdot 147 + 147$ | 8,526 |
| `h2h = Linear(147, 147)` | $147 \cdot 147 + 147$ | 21,756 |
| `h2o = Linear(147, 18)` | $147 \cdot 18 + 18$ | 2,664 |
| **Total** | | **32,946** |

#### LSTM, $n\_h = 64$

| Componente | Cálculo | Parámetros |
|------------|---------|-----------:|
| `LSTMCell(57, 64)` weight_ih | $4 \cdot 64 \cdot 57$ | 14,592 |
| `LSTMCell(57, 64)` weight_hh | $4 \cdot 64 \cdot 64$ | 16,384 |
| `LSTMCell(57, 64)` biases (ih + hh) | $4 \cdot 64 + 4 \cdot 64$ | 512 |
| `h2o = Linear(64, 18)` | $64 \cdot 18 + 18$ | 1,170 |
| **Total** | | **32,658** |

#### BiLSTM, $n\_h = 40$

| Componente | Cálculo | Parámetros |
|------------|---------|-----------:|
| `forward_cell = LSTMCell(57, 40)` | $4 \cdot 40 \cdot 57 + 4 \cdot 40 \cdot 40 + 8 \cdot 40$ | 15,840 |
| `backward_cell = LSTMCell(57, 40)` | (idem) | 15,840 |
| `h2o = Linear(80, 18)` | $80 \cdot 18 + 18$ | 1,458 |
| **Total** | | **33,138** |

### Comparación

| Modelo | $n\_hidden$ | Parámetros |
|--------|------------:|-----------:|
| RNN vanilla | 147 | **32,946** |
| LSTM | 64 | **32,658** |
| BiLSTM | 40 | **33,138** |

Los tres modelos están en **~33K parámetros** (diferencia de menos del 1.5% entre el menor y el mayor). Las dimensiones ocultas (147 / 64 / 40) fueron elegidas a propósito para llegar a esta paridad de capacidad.

### Por qué no sería justo usar el mismo $n\_hidden$

Si fijáramos $n\_hidden = 64$ para los tres modelos:

| Modelo | $n\_hidden = 64$ | Parámetros |
|--------|----------------:|-----------:|
| RNN vanilla | 64 | **9,042** |
| LSTM | 64 | **32,658** |
| BiLSTM | 64 | **65,298** |

La BiLSTM tendría **~7×** más parámetros que la RNN vanilla. La razón es estructural:

- La **RNN vanilla** tiene 1 transformación recurrente: $W_{hh} \in \mathbb{R}^{H \times H}$.
- La **LSTM** tiene 4 transformaciones recurrentes (input, forget, candidate, output): coeficiente **×4** sobre la matriz $W_{hh}$.
- La **BiLSTM** tiene 2 LSTMs (forward + backward): coeficiente adicional **×2** → en total **×8** respecto a la RNN vanilla.

Comparar las tres arquitecturas con el mismo $n\_hidden$ confunde dos efectos: la **diferencia de arquitectura** (gating, bidireccionalidad) con la **diferencia de capacidad** (número de parámetros). Si la BiLSTM le gana a la RNN vanilla, no sabemos si es porque la bidireccionalidad ayuda o porque tiene 7× más parámetros para memorizar.

Igualar el número de parámetros (vía $n\_hidden$ asimétrico) **aísla** la contribución de la arquitectura. Es la práctica estándar al comparar familias de modelos recurrentes.

---

## Actividad 2 — Clasificador con capa oculta + ReLU

### Enunciado

Reemplazar el clasificador del LSTM (no bidireccional, $n\_hidden = 64$) por un MLP de dos capas con ReLU intermedia:

```
h_lstm  →  Linear(64, mlp_hidden)  →  ReLU  →  Linear(mlp_hidden, 18)  →  logits
```

Probar `mlp_hidden ∈ {50, 150, 300}`. Graficar la pérdida y comentar la cantidad de parámetros.

### Implementación

```python
class LSTM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_hidden, output_size):
        super().__init__()
        self.bidirectional = False
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.h2o = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_size),
        )

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.cell(input, hidden)
        output = self.h2o(hidden[0])
        return output, hidden
```

### Conteo de parámetros

El backbone LSTM (cell con $n\_hidden=64$) aporta siempre **31,488** parámetros. Solo cambia el head:

| Variante | Head: `Linear(64, mh) + Linear(mh, 18)` | Head | Total |
|----------|----------------------------------------:|-----:|------:|
| LSTM original (head simple) | $64 \cdot 18 + 18$ | 1,170 | **32,658** |
| LSTM + MLP head, $mh = 50$ | $64 \cdot 50 + 50 + 50 \cdot 18 + 18$ | 4,168 | **35,656** |
| LSTM + MLP head, $mh = 150$ | $64 \cdot 150 + 150 + 150 \cdot 18 + 18$ | 12,468 | **43,956** |
| LSTM + MLP head, $mh = 300$ | $64 \cdot 300 + 300 + 300 \cdot 18 + 18$ | 24,918 | **56,406** |

**Observaciones sobre el conteo:**

- El head crece **linealmente** con `mlp_hidden`: cada neurona añade ~83 parámetros (64 entrada + 18 salida + 1 bias en cada lado).
- La variante con $mh=300$ casi **duplica** el tamaño del modelo respecto al original (56K vs 33K). El gasto extra está casi todo en el clasificador, no en el backbone recurrente.
- El backbone LSTM (donde realmente ocurre el aprendizaje secuencial) **no cambia**. Lo único que se está variando es la capacidad del clasificador final que recibe el último estado oculto.

### Curvas de pérdida — resultados reales

Las tres variantes se entrenaron durante 200,000 iteraciones cada una (batch=1, lr=0.005, sin gradient clipping). Tiempos de entrenamiento prácticamente iguales (~9 min cada uno) porque el cómputo dominante es el LSTM backbone, que no cambia entre las variantes.

#### LSTM + MLP head, $mh = 50$ (35,656 parámetros)

```text
10000   5%  (0m 27s)  2.9321  Araullo / French         ✗ (Portuguese)
30000  15%  (1m 22s)  2.7806  Sztegon / Dutch          ✗ (Czech)
50000  25%  (2m 17s)  1.2006  Ta / Chinese             ✗ (Vietnamese)
80000  40%  (3m 39s)  1.1625  Finnegan / Irish         ✓
100000 50%  (4m 35s)  0.0650  Sklavenitis / Greek      ✓
130000 65%  (5m 58s)  0.2613  Sook / Korean            ✓
170000 85%  (7m 48s)  0.3171  Fei / Chinese            ✓
200000 100% (9m 10s)  3.7960  Styles / Greek           ✗ (English)
```

![Curva de pérdida LSTM + MLP head, mh=50](/laboratorios/lab-11/loss-act2-mh50.png)

Plateau plano hasta iter ~25K, caída brusca entre iter 25K–50K, descenso gradual hasta **~0.95–1.00 al final**.

#### LSTM + MLP head, $mh = 150$ (43,956 parámetros)

```text
10000   5%  (0m 27s)  2.9794  Losa / German            ✗ (Spanish)
30000  15%  (1m 24s)  2.4939  Antwerp / Scottish       ✗ (Dutch)
60000  30%  (2m 46s)  0.9761  Esteves / Portuguese     ✓
100000 50%  (4m 36s)  1.2488  Tze / Chinese            ✓
130000 65%  (5m 58s)  0.0505  O'Dell / Irish           ✓
170000 85%  (7m 49s)  0.9912  Flores / Spanish         ✓
200000 100% (9m 11s)  0.1196  Calpis / Greek           ✓
```

![Curva de pérdida LSTM + MLP head, mh=150](/laboratorios/lab-11/loss-act2-mh150.png)

Plateau más corto (~18–20K iters), descenso continuo desde ahí sin escalón pronunciado, hasta **~0.85–0.90 al final**.

#### LSTM + MLP head, $mh = 300$ (56,406 parámetros)

```text
10000   5%  (0m 28s)  2.8101  Ishimura / Vietnamese    ✗ (Japanese)
30000  15%  (1m 25s)  1.5157  Bobienski / Polish       ✓
50000  25%  (2m 22s)  1.7629  Desmond / Dutch          ✗ (Irish)
70000  35%  (3m 19s)  0.0551  Takizawa / Japanese      ✓
110000 55%  (5m 9s)   0.0570  Sniegowski / Polish      ✓
150000 75%  (7m 0s)   0.9829  Watt / Scottish          ✓
180000 90%  (8m 25s)  0.0343  Krakowski / Polish       ✓
200000 100% (9m 21s)  0.6386  Souza / Portuguese       ✓
```

![Curva de pérdida LSTM + MLP head, mh=300](/laboratorios/lab-11/loss-act2-mh300.png)

Plateau corto (~18–20K iters), caída más pronunciada (de 2.85 a ~1.8 entre iter 20K–35K), descenso continuo hasta **~0.75–0.85 al final** — el más bajo de las tres variantes.

### Comentario sobre las curvas

| Variante | Params | Plateau dura | Loss final (suavizada) |
|----------|-------:|:------------:|:----------------------:|
| $mh = 50$ | 35,656 | ~25K iters | ~0.95–1.00 |
| $mh = 150$ | 43,956 | ~18–20K iters | ~0.85–0.90 |
| $mh = 300$ | 56,406 | ~18–20K iters | **~0.75–0.85** ← menor |

Tres observaciones empíricas:

1. **El head más grande arranca antes a aprender.** El primer ✓ confiable aparece en iter ~30K para $mh=300$ (Bobienski / Polish), iter ~60K para $mh=150$ (Esteves / Portuguese), iter ~80K para $mh=50$ (Finnegan / Irish). Más capacidad en el clasificador permite encontrar más rápido una frontera de decisión que separa las 18 clases.
2. **Existe una mejora pequeña pero consistente al aumentar $mlp\_hidden$.** La diferencia es de ~0.20 puntos de loss entre el extremo ($mh=50$) y el más grande ($mh=300$). No es dramática pero es real.
3. **Los tiempos de entrenamiento son casi idénticos (~9 min).** El cómputo dominante es el LSTMCell (que no cambia), no el head MLP. El head extra aporta menos del 1% del tiempo de cómputo.

### Comentario sobre los parámetros

El backbone LSTM (LSTMCell con $n\_hidden=64$) aporta siempre 31,488 parámetros. Solo cambia el head:

- El head crece **linealmente** con $mlp\_hidden$: cada neurona añade aproximadamente 83 parámetros (64 entradas + 18 salidas + 2 sesgos).
- La variante con $mh=300$ casi **duplica** el tamaño del modelo respecto al LSTM original (56K vs 33K). Casi todo ese gasto extra está en el clasificador, no en el backbone recurrente.
- El backbone LSTM (donde realmente ocurre el aprendizaje secuencial) **no cambia** entre las variantes.

### Conclusión

Aumentar la capacidad del clasificador final mejora la pérdida final, pero la **ganancia por parámetro extra es marginal**: pasar de 33K parámetros (LSTM original) a 56K parámetros ($mh=300$) compra una mejora de ~0.10–0.20 puntos de loss. La mayor parte del trabajo discriminativo lo hace el LSTM, no el clasificador.

**El cuello de botella** está en la representación que entrega el LSTM — el último estado oculto $h_T \in \mathbb{R}^{64}$. Mientras esa representación sea la limitación, hacer el clasificador más grande es ineficiente. Para mejorar el modelo seriamente conviene primero subir $n\_hidden$ del LSTM (más capacidad por step), no agrandar el head.

Esto valida una intuición común en arquitecturas profundas: **el costoso es el backbone, no la cabeza**. Los heads MLP extra son útiles cuando el backbone ya está saturado y queda capacidad sin uso, pero en este lab el LSTM con $n\_hidden=64$ aún no está aprovechado al límite.
