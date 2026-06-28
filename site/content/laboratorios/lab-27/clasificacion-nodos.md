---
title: "Clasificación de nodos: el experimento que prueba el valor del grafo"
weight: 2
---

El primer nivel de tarea en grafos es **clasificar nodos dentro de un grafo**. El lab lo aborda en dos pasos: un ejemplo de juguete (Karate Club) que revela el *inductive bias* de las GCN, y un experimento controlado (Cora) que **mide** cuánto aporta la estructura.

## La capa GCN de Kipf & Welling (2017)

Toda la sección descansa en esta ecuación:

$$
\mathbf{h}_i^{(t+1)} = \mathbf{W}^{(t+1)} \sum_{j \in \mathcal{N}(i)\,\cup\,\{i\}} \frac{1}{c_{i,j}} \, \mathbf{h}_j^{(t)}
$$

Pieza por pieza:

- **$\mathbf{h}_j^{(t)}$** — embedding del nodo $j$ en la capa $t$ (en $t=0$ es el feature $\mathbf{x}_j$).
- **$\sum_{j \in \mathcal{N}(i)\cup\{i\}}$** — suma sobre los **vecinos de $i$ más el propio $i$**. Ese "$\cup\,\{i\}$" es el **self-loop**: sin él, un nodo olvidaría su propia información al actualizarse.
- **$\frac{1}{c_{i,j}}$** — normalización **fija, no entrenable**: $c_{i,j} = \sqrt{\deg(i)}\sqrt{\deg(j)}$. Evita que nodos con muchos vecinos exploten en magnitud.
- **$\mathbf{W}^{(t+1)}$** — la **única parte entrenable**, compartida por todos los nodos.

> En una frase: *cada nodo actualiza su estado como un promedio normalizado de los estados de sus vecinos (y el suyo), seguido de una transformación lineal aprendida.* Esto es **message passing**. Apilar $T$ capas hace que la información viaje hasta $T$ saltos — la base de la [Actividad 6](actividades).

**Analogía con una CNN:** una convolución de imagen agrega sobre una vecindad fija (kernel 3×3, 9 píxeles en grilla regular); una GCN agrega sobre una vecindad **irregular y de tamaño variable** definida por las aristas. El peso $\mathbf{W}$ se comparte entre todos los nodos igual que el kernel entre todos los píxeles — de ahí "convolución".

## Karate Club: estructura **sin entrenar**

```python
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = GCNConv(dataset.num_features, 4)   # 34 → 4
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)                       # → 2D para graficar
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).tanh()
        h = self.conv2(h, edge_index).tanh()
        h = self.conv3(h, edge_index).tanh()
        return self.classifier(h), h                    # logits, embedding 2D
```

El embudo `34 → 4 → 4 → 2` comprime a 2 dimensiones **a propósito**, para graficar el embedding sin TSNE. Lo notable ocurre al visualizar el modelo **recién instanciado, sin entrenar** (pesos aleatorios):

![Embedding GCN de Karate Club sin entrenar — ya agrupa por comunidad](/images/lab27-c30.png)

Los nodos **ya aparecen agrupados por comunidad**. ¿Cómo, si $\mathbf{W}$ es aleatorio? Aunque los pesos no aprendieron nada, la operación de **agregar sobre vecinos** mezcla los estados: dos nodos que comparten vecinos reciben sumas parecidas y, tras la misma $\mathbf{W}$ compartida, salen con embeddings parecidos. La señal viene de la **topología procesada por la operatoria de la GCN** — no de los features (one-hot, inútiles) ni de los pesos (aleatorios). Este es el **inductive bias** de homofilia, y es exactamente la respuesta de la [Actividad 1](actividades).

### Entrenamiento

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = criterion(out[data.train_mask], data.y[data.train_mask])   # solo nodos enmascarados
```

Karate Club es **clasificación semi-supervisada de nodos**: solo unos pocos nodos (los "líderes") tienen label durante el entrenamiento, pero el forward pass propaga por **todo** el grafo. La GCN extiende esas pocas etiquetas al resto vía la estructura — el rasgo que hace tan poderosas a las GNN. El GIF del entrenamiento (800 épocas) muestra cómo los clusters se separan progresivamente:

![GIF del entrenamiento de la GCN en Karate Club](/images/lab27-c32.gif)

La separación no es perfecta: los nodos "puente" (conectados a ambas facciones) son ambiguos por su propia conectividad. Una limitación honesta y esperable.

## Cora: el experimento controlado MLP vs GCN

Aquí está la jugada experimental del lab. **Cora** es una red de citas: 2.708 papers (nodos), citas (aristas), features bag-of-words de 1.433 dims, 7 categorías temáticas.

```python
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
```

`NormalizeFeatures()` normaliza cada vector bag-of-words por fila (suma 1), estándar para evitar que papers largos dominen por magnitud. El nombre **Planetoid** viene de Yang et al. (2016), que estandarizó los splits de Cora/CiteSeer/PubMed.

### Baseline: un MLP que **ignora el grafo**

```python
class MLP(torch.nn.Module):
    def forward(self, x):              # ← solo x, NO edge_index
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)
```

El MLP clasifica cada paper **solo por su bag-of-words**, tratándolo como un punto tabular aislado. La estructura de citas le es **invisible**. Con `dropout=0.5` y `weight_decay=5e-4` (regularización fuerte, porque solo hay 140 nodos de train y 1.433 features):

```text
Test Accuracy: 0.5740
```

### La GCN: **misma arquitectura, ahora con el grafo**

La comparación es deliberadamente justa: dos capas, mismo `hidden_channels=16`, mismo dropout, mismo seed `12345`, mismo optimizador. La **única** diferencia:

```python
class GCN(torch.nn.Module):
    def forward(self, x, edge_index):     # ← ahora SÍ recibe edge_index
        x = self.conv1(x, edge_index).relu()      # GCNConv en vez de Linear
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
```

Cambiamos `Linear` → `GCNConv`. Eso es todo. Ahora cada paper, al clasificarse, agrega información de los papers que cita y lo citan — explotando la homofilia (*los papers que se citan tienden a ser del mismo tema*).

```text
Test Accuracy: 0.8130
```

| Modelo | Usa el grafo | Test Acc |
|---|---|---|
| MLP | ❌ | 57.4% |
| GCN | ✅ | **81.3%** |

**+24 puntos absolutos** con la misma capacidad de parámetros y los mismos features. Lo único que cambió fue *dejar que el modelo mire el grafo*. El contraste visual lo confirma — antes de entrenar (TSNE, clases mezcladas) vs después (siete clusters limpios):

![Cora GCN antes de entrenar](/images/lab27-c44.png)

![Cora GCN después de entrenar — siete clusters separados](/images/lab27-c48.png)

{{< callout type="info" >}}
**El experimento controlado como pedagogía.** Notá la disciplina: cambiar **una sola variable** (usar o no `edge_index`) y congelar todo lo demás. Es exactamente cómo aislarías el efecto de una señal en cualquier sistema —incluido un pipeline de *record linkage*: si querés saber cuánto aporta una feature, congelá el resto y medí el delta. Acá el delta es +24 puntos, atribuibles enteramente a la estructura del grafo.
{{< /callout >}}

---

Hasta aquí clasificamos nodos *dentro* de un grafo. La siguiente sección cambia el paradigma: clasificar **grafos enteros**. Sigue en [clasificación de grafos](clasificacion-grafos).
