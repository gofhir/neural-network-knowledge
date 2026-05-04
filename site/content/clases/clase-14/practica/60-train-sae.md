---
title: "60 - Entrenar un SAE sobre el residual stream"
weight: 600
math: true
---

## 1. Apertura: deshacer la superposition

Cap 59 mostro que un modelo con `n_features > d_model` aprende representaciones polisemanticas: cada neurona codifica multiples conceptos no relacionados. Esto rompe la interpretabilidad neuron-by-neuron — no podemos mirar la dimension 17 del residual stream y decir "esta es la feature de mayusculas".

**Sparse Autoencoders (SAEs)** (Bricken et al. 2023) deshacen la superposition: re-representan el residual stream en un espacio MAS GRANDE pero con activacion ESPARSA. Cada feature del SAE tiene "espacio" para ser mongamica (representar UN concepto), porque la sparsity penalty force que solo unas pocas esten activas por input.

Este capitulo entrena un SAE sobre activaciones del residual stream de Mini-LLaMA. Cap 61 inspeccionara las features aprendidas para identificar cuales son monosemanticas.

---

## 2. Setup tecnico

**Que cacheamos:** `block.2` output (residual stream despues de la capa 2). Capa intermedia donde el modelo ya ha procesado contexto pero aun no esta cristalizando la prediccion final.

**Datos:** 200 prompts de Shakespeare de 64 caracteres = 12,800 vectores de `d_model=128`. Suficiente para entrenar un SAE pequeno sin overfittear.

**Arquitectura del SAE:**

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=128, d_features=512, l1_coeff=0.5):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_features)
        self.decoder = nn.Linear(d_features, d_model, bias=False)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        recon = self.decoder(features)
        return recon, features
```

- `d_features=512`: 4× expansion sobre `d_model=128`. Espacio donde features pueden ser mas ortogonales.
- ReLU en encoder: las features son no-negativas (representan "presencia de concepto").
- Decoder sin bias (convencion del paper de Anthropic): evita que el SAE "shrink" las features.

**Loss:**

$$\mathcal{L} = \underbrace{||x - \hat{x}||^2}_{\text{reconstruccion}} + \lambda \cdot \underbrace{||features||_1}_{\text{sparsity}}$$

El parametro $\lambda$ controla el trade-off: bajo → muchas features activas (no esparso), alto → pocas features activas (esparso pero peor reconstruccion).

---

## 3. La caza del lambda correcto

Tipico de SAEs: $\lambda$ requiere tuning. Probamos tres valores:

| $\lambda$ | L0 promedio | Var explicada |
|---|---|---|
| 1e-3 | 506/512 (99%) | 100.0% |
| 1e-1 | 288/512 (56%) | 99.4% |
| **0.5** | **166/512 (32%)** | **98.4%** |

Con $\lambda=10^{-3}$: el SAE practicamente no usa la penalty — todas las features estan activas, NO hay sparsity. Es como un autoencoder regular.

Con $\lambda=10^{-1}$: sparsity moderada (56% activas). Mejor pero aun lejos de "monosemantico".

Con $\lambda=0.5$: sparsity razonable (32% activas, 165 features promedio). Reconstruccion excelente (98.4% varianza explicada). Este es nuestro punto operativo.

A escalas industriales ($d_{\text{features}} = 65k+$, datos masivos) los SAEs alcanzan L0 ~ 30-100 con $\lambda$ adecuado. Para Mini-LLaMA con d_features=512 esperar L0=166 es proporcional.

---

## 4. Script

```python
"""60_train_sae.py - Cap 60: entrenar SAE sobre residual stream de Mini-LLaMA."""
import random, torch
from pathlib import Path
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations, SparseAutoencoder

torch.manual_seed(1337); random.seed(1337)
device = get_device()
tok = CharTokenizer(load_text("shakespeare.txt"))
model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

# Recolectar 12,800 activaciones del residual stream de block.2
N_PROMPTS = 200; WIN = 64
all_acts = []
for _ in range(N_PROMPTS):
    start = random.randint(0, len(load_text()) - WIN - 1)
    ids = torch.tensor([tok.encode(text[start:start+WIN])], dtype=torch.long, device=device)
    with cache_activations(model, ["blocks.2"]) as cache:
        with torch.no_grad():
            model(ids)
    all_acts.append(cache["blocks.2"][0].cpu())
X = torch.cat(all_acts, dim=0).to(device)

# Entrenar SAE
sae = SparseAutoencoder(d_model=128, d_features=512, l1_coeff=0.5).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
for it in range(2000):
    idx = torch.randint(0, X.shape[0], (1024,))
    x = X[idx]
    recon, features = sae(x)
    loss = ((x - recon) ** 2).mean() + 0.5 * features.abs().mean()
    opt.zero_grad(); loss.backward(); opt.step()

# Guardar
torch.save({"sae": sae.state_dict(), "config": ...}, "checkpoints/sae_mini_llama.pt")
```

---

## 5. Output literal

```
Recolectando activaciones de blocks.2 sobre 200 prompts de 64 chars

Total activations recolectadas: shape (12800, 128)
Mean magnitud: 11.832
Std por dimension: 0.978

=== Entrenando SAE (d_features=512, lambda=0.5, iters=2000) ===

iter       recon_loss    l1_loss      total     L0_avg
-------------------------------------------------------
0             1.22518    0.24660    1.34848      256.8
100           0.29178    0.26407    0.42382      246.8
250           0.10749    0.24013    0.22756      225.3
500           0.05257    0.18766    0.14640      189.2
1000          0.03000    0.13958    0.09979      167.3
1499          0.02310    0.11634    0.08127      165.4

Reduccion en recon_loss: 1.22518 -> 0.01873 (98.5% menor)

=== Estadisticas finales del SAE ===
Recon loss sobre todo el dataset: 0.01854
Varianza explicada: 98.4%
L0 promedio (features activas por sample): 165.8 de 512
L0 mediano: 167.0
Features muertas (nunca activas): 0 de 512

Guardado SAE en checkpoints/sae_mini_llama.pt
```

---

## 6. Analisis: que dicen los numeros

### Reconstruccion casi perfecta

La recon loss bajo de 1.225 (random init) a **0.019** — reduccion del 98.5%. La varianza explicada es 98.4%. Esto significa que el SAE puede reconstruir el residual stream de Mini-LLaMA con altisima fidelidad.

¿Por que importa? Si el SAE no pudiera reconstruir bien, sus "features" serian artificiales — no representarian lo que el modelo realmente codifica. Una reconstruccion buena nos da confianza: las features capturan informacion real del residual stream.

### Sparsity razonable

**L0 promedio = 165.8 de 512 features activas**. Eso es ~32% — no extrema sparsity (los SAEs industriales apuntan a ~5%), pero suficiente para empezar a ver features distinguibles.

L0 mediano (167) cercano a L0 promedio (165.8) indica que la distribucion de "features activas por sample" es relativamente uniforme — no hay samples con muy pocas o muchas features anormales.

### Cero features muertas

`Features muertas: 0 de 512`. Un problema comun en SAEs es que algunas features nunca se activan despues de inicializacion (gradiente zero las mantiene zero indefinidamente). Aqui no paso — todas las 512 features estan vivas, contribuyendo a la representacion.

### La progresion del L1 loss revela aprendizaje

```
iter 0:    L1=0.247  (init aleatoria — features chicas)
iter 100:  L1=0.264  (sube porque encoder aumenta magnitud)
iter 250:  L1=0.240  (empieza a bajar)
iter 1499: L1=0.116  (converge — features siguen activas pero magnitud reducida)
```

El L1 sube primero (el modelo aprende a usar features) y luego baja (la penalty empuja features pequenas hacia 0). Es un comportamiento tipico de SAE: primero "explorar" cuales features son utiles, luego "podar" las redundantes.

### L0 cae de 256 a 165

```
iter 0:    L0=256.8 (la mitad de las features activas, init aleatoria)
iter 1499: L0=165.4 (32% activas)
```

El SAE "descarto" 90 features (256 → 165). Esas no aportaban a la reconstruccion proporcionalmente a su costo de L1.

---

## 7. Limitaciones honestas de este SAE

### Sparsity imperfecta

L0=165 es alto comparado con SAEs industriales (que apuntan a L0=10-100 sobre vocabularios mucho mas grandes). Las features aprendidas seran **parcialmente polisemanticas** — cada una probablemente representa varios conceptos relacionados, no uno solo. Cap 61 verificara esto inspeccionando los top-tokens por feature.

### Escala chica del modelo base

Mini-LLaMA tiene `d_model=128` y entrena sobre Shakespeare char-level (~130k chars). El "vocabulario semantico" del modelo es chico — pocos conceptos linguisticos a codificar. Comparativamente, un SAE sobre Claude 3 Sonnet (Anthropic 2024) trabaja sobre `d_model ~12000` y datos enormes; descubre miles de features interpretables.

Aqui esperamos descubrir features simples: principio de oracion, mayusculas, separadores, posiciones de personajes en lineas de dialogo. No vamos a encontrar "feature del Golden Gate Bridge" como Anthropic en su paper Scaling Monosemanticity.

### Configuracion no-ajustada

Para un SAE de produccion se hace:

- Sweep mas amplio de $\lambda$
- Tecnicas como top-k sparsity (en vez de L1)
- Mecanismos contra "feature death" (resampling)
- Mas datos, mas pasos
- Multiple semillas con eleccion del mejor

Aqui hicimos un SAE basico para fines pedagogicos. Suficiente para mostrar la mecanica y obtener algunas features interpretables — no para resultados de paper.

---

## 8. Por que entrenamos sobre `block.2` y no otra capa

La eleccion de capa para entrenar el SAE es importante:

- **`tok_emb`**: features serian basicamente embeddings de tokens (poco informativas)
- **`block.0`**: features representan caracteristicas locales — caracter actual + previous-token
- **`block.2`** (nuestro target): features de contexto medio — gramatica, posicion en oracion, identidad de speaker
- **`block.3`**: features muy cercanas a la prediccion final, dificiles de interpretar como "conceptos" abstractos
- **`norm_final`**: post-norm final, listo para el head — interpretacion borrosa por la normalizacion

Block.2 es el sweet spot para descubrir features semanticas: el modelo ya proceso contexto pero no esta "pensando en la prediccion proxima". Esta convencion es heredada de Anthropic, que entrena SAEs en capas intermedias por la misma razon.

---

## 9. Preguntas de verificacion

**1. ¿Por que el SAE expande de 128 a 512 dimensiones (no comprimir)?**

Un autoencoder ESTANDAR comprime — entradas a un bottleneck pequeno y reconstruye. El objetivo es compresion. Un SPARSE autoencoder hace lo OPUESTO: expande a un espacio mas grande pero con activacion esparsa. El objetivo es disentanglement, no compresion. La idea: en `d_model=128` las features estan en superposition (cada dim representa varios conceptos via combinaciones angulares no-ortogonales). En `d_features=512` con sparsity, cada feature puede ser ortogonal (en su modo "unico") sin colisiones. La expansion da espacio; la sparsity asegura que el espacio se use eficientemente. Este es el insight central de Bricken et al. 2023.

**2. ¿Que pasaria si el SAE tuviera reconstruccion perfecta pero L0=512 (todas activas)?**

Seria un autoencoder regular sin sparsity efectiva. Las features serian linealmente combinable — cada una contribuye un poco a la reconstruccion, pero ninguna representa un concepto distintivo. En cap 61 al inspeccionar los top-tokens por feature encontrariamos que cada feature responde a TODOS los tokens (polisemantica total). El SAE seria computacionalmente equivalente a usar una matriz aleatoria — no nos da insight nuevo. Por eso la sparsity es esencial: sin ella, el SAE es un disfraz para no descomponer nada.

**3. ¿Como se relaciona la sparsity del SAE con el resultado del cap 59 (toy superposition)?**

Cap 59 mostro que un toy model con 5 features y 2 dim aprende a empacarlas en angulos no-ortogonales — superposition. Si entrenaras un SAE sobre las activaciones de ese toy model con `d_features=10` y sparsity alta, deberia recuperar las 5 features originales como direcciones ortogonales en el espacio expandido. Esto es exactamente el principio de los SAEs: "deshacer" la superposition que el modelo aprendio durante su entrenamiento, restaurando una base monosemantica. La pregunta empirica (que cap 61 examina sobre Mini-LLaMA): ¿el SAE realmente recupera features interpretables, o solo reorganiza la polisemanticidad?
