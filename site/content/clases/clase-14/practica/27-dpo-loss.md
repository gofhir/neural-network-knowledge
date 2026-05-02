---
title: "27 - DPO loss: la derivacion paso a paso"
weight: 270
math: true
---

En [cap 26](../26-preferencias-bradley-terry) vimos que Bradley-Terry modela preferencias con `P(y_w succeq y_l) = sigma(r_w - r_l)`. RLHF clasico aprende ese reward `r` con un MLP separado y despues hace PPO. DPO se salta ese paso. ¿Como? La derivacion. En [cap 28](../28-dpo-dataset) construiremos el dataset de pares; aqui construimos la pieza matematica que justifica todo lo siguiente.

---

## 1. La idea central

La intuicion del paper de Rafailov 2023: en el optimo del problema de RLHF (con KL implicito al ref model), existe una **forma cerrada** de la policy optima en terminos del reward. Si invertimos esa relacion, podemos expresar el reward EN TERMINOS DE la policy y el ref. Y eso lo metemos directamente en la loss Bradley-Terry.

Resultado: una loss que opera SOBRE la policy, no sobre un reward model. Una sola fase de training, no tres. Sin reward model que entrenar, sin PPO, sin sampling on-policy. Solo el dataset de pares y dos copias del SFT (una congelada, otra que aprende).

---

## 2. La formula completa

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

Donde:

- `pi_theta` = policy (la que entrenamos).
- `pi_ref` = reference (el modelo SFT, congelado).
- `beta` = hiperparametro (KL strength).
- `(x, y_w, y_l)` = triple del dataset DPO (prompt, chosen, rejected).
- `sigma` = sigmoide.

Nota importante: `pi_theta` y `pi_ref` se inicializan al MISMO modelo (el SFT). `pi_ref` se congela (no recibe gradiente). `pi_theta` es lo que aprende. Al inicio de DPO, las dos son identicas — y veremos numericamente que eso implica `loss = log 2`.

---

## 3. Que hace cada pieza

**Los log-ratios.** El termino $\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ es la diferencia logaritmica entre lo que la policy actual asigna y lo que el ref asigna a la misma respuesta. Si `pi_theta == pi_ref` (al inicio), el ratio es 0. Si la policy se desvia y aumenta la prob de `y`, el ratio se hace positivo. Si la baja, negativo. Es un detector de cuanto se ha movido la policy del SFT en cada respuesta concreta.

**El KL implicito.** Estos log-ratios actuan como un KL implicito al reference. La policy NO es libre de irse a cualquier distribucion — esta penalizada por alejarse demasiado del SFT. Esto evita el `mode collapse` clasico de RL: que la policy degenere a generar la misma respuesta para todo. La derivacion de Rafailov muestra que minimizar la loss DPO es equivalente a maximizar `E[r] - beta * KL(pi_theta || pi_ref)` con un reward implicito — el KL no aparece explicitamente en la loss pero sus efectos si.

**Beta — la regulizacion.** `beta` controla cuanto pesa el KL. `beta=0.1` (el valor default y el que usamos): conservador, la policy se desvia poco. `beta=1.0`: agresivo, casi sin freno. `beta` muy chico: el modelo casi no aprende. `beta` muy grande: el modelo destruye su SFT y colapsa a una distribucion degenerada que solo favorece chosen sobre rejected en los ejemplos del dataset, sin generalizar.

**`log pi(y|x)` numericamente.** Para una respuesta `y = (y_1, ..., y_R)` dada un prompt `x`,

$$
\log \pi(y \mid x) = \sum_{t=1}^{R} \log \pi(y_t \mid x, y_1, \dots, y_{t-1}).
$$

Es decir, la suma de log-probs token a token, condicionando en todo lo anterior. Esto es exactamente lo que `compute_logp_response` calcula en `_models.py`. Para una respuesta de R tokens, hacemos un solo forward pass del modelo sobre el prompt + respuesta, leemos los logits en las posiciones que predicen cada `y_t`, aplicamos log_softmax, y sumamos los R valores que corresponden al token correcto.

---

## 4. El script

```python
"""19_dpo_loss_derivation.py - Cap 27: DPO loss paso a paso para 1 triple.

Verifica que `dpo_loss` del modulo es coherente con calculo manual.
Al iniciar DPO con policy=ref=SFT, log-ratios son 0 y loss=-log(0.5).
"""
import torch
from _models import load_pretrained_mini_llama, compute_logp_response, dpo_loss
from _eval import build_char_maps

torch.manual_seed(1337)
text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
for p in ref.parameters():
    p.requires_grad_(False)

prompt = "INSTR: reverse 'cat'\nRESP: "
chosen = "tac\n"
rejected = "CAT\n"
beta = 0.1

p_ids = torch.tensor([c2i[c] for c in prompt],   dtype=torch.long)
c_ids = torch.tensor([c2i[c] for c in chosen],   dtype=torch.long)
r_ids = torch.tensor([c2i[c] for c in rejected], dtype=torch.long)

print("=== DPO loss paso a paso ===\n")
print(f"Prompt:   {prompt!r}")
print(f"Chosen:   {chosen!r}")
print(f"Rejected: {rejected!r}")
print(f"Beta:     {beta}\n")

logp_pi_w  = compute_logp_response(policy, p_ids, c_ids)
logp_pi_l  = compute_logp_response(policy, p_ids, r_ids)
logp_ref_w = compute_logp_response(ref,    p_ids, c_ids)
logp_ref_l = compute_logp_response(ref,    p_ids, r_ids)

print(f"log pi_theta(y_w|x)  = {logp_pi_w.item():+.4f}")
print(f"log pi_theta(y_l|x)  = {logp_pi_l.item():+.4f}")
print(f"log pi_ref(y_w|x)    = {logp_ref_w.item():+.4f}")
print(f"log pi_ref(y_l|x)    = {logp_ref_l.item():+.4f}")

ratio_w = logp_pi_w - logp_ref_w
ratio_l = logp_pi_l - logp_ref_l
print(f"\nlog ratio chosen   = {ratio_w.item():+.4f}")
print(f"log ratio rejected = {ratio_l.item():+.4f}")

z = beta * (ratio_w - ratio_l)
loss_manual = -torch.nn.functional.logsigmoid(z)
print(f"\nbeta*(ratio_w - ratio_l) = {z.item():+.4f}")
print(f"loss_manual = -log sigmoid(z) = {loss_manual.item():.4f}")

loss_helper = dpo_loss(policy, ref, p_ids, c_ids, r_ids, beta=beta)
print(f"loss_helper                  = {loss_helper.item():.4f}")
assert abs(loss_manual.item() - loss_helper.item()) < 1e-4, "helper mismatch with manual"
print("\nOK: helper coincide con calculo manual.")
print("\nAl iniciar DPO desde SFT, policy=ref => ratios=0 => loss=-log(0.5)=0.6931.")
```

---

## 5. Output

```
=== DPO loss paso a paso ===

Prompt:   "INSTR: reverse 'cat'\nRESP: "
Chosen:   'tac\n'
Rejected: 'CAT\n'
Beta:     0.1

log pi_theta(y_w|x)  = -3.4508
log pi_theta(y_l|x)  = -21.6548
log pi_ref(y_w|x)    = -3.4508
log pi_ref(y_l|x)    = -21.6548

log ratio chosen   = +0.0000
log ratio rejected = +0.0000

beta*(ratio_w - ratio_l) = +0.0000
loss_manual = -log sigmoid(z) = 0.6931
loss_helper                  = 0.6931

OK: helper coincide con calculo manual.

Al iniciar DPO desde SFT, policy=ref => ratios=0 => loss=-log(0.5)=0.6931.
```

---

## 6. Lectura del output — interpretar los numeros

**El log-prob asimetrico.** Mira los log-probs:

- `log pi(y_w='tac\n' | prompt) = -3.4508`
- `log pi(y_l='CAT\n' | prompt) = -21.6548`

La policy SFT le asigna a `'tac'` una probabilidad de $e^{-3.45} \approx 3.2\%$. A `'CAT'`, $e^{-21.65} \approx 4 \times 10^{-10}$. La diferencia es ENORME — el SFT aprendio fuerte a hacer reverse, no upper, sobre prompts de reverse. Es la confirmacion cuantitativa de lo que vimos cualitativamente en [cap 25](../25-sft-eval): el SFT ya prefiere claramente la respuesta "correcta de tarea" sobre la "correcta de otra tarea". DPO no parte de cero — parte de un modelo que ya sabe distinguir, y refina esa distincion.

**Los ratios son 0.** Tanto el ratio chosen como rejected son `+0.0000`. ¿Por que? Porque `pi_theta` y `pi_ref` son el MISMO modelo (los dos cargan `mini_llama_sft.pt`). Sin entrenamiento, los logs son identicos, las restas son cero, y `ratio_w = ratio_l = 0`. Esto NO es una coincidencia — es la condicion inicial obligatoria de cualquier corrida DPO bien hecha.

**La loss es 0.6931.** $z = \beta \cdot (\text{ratio}_w - \text{ratio}_l) = 0.1 \cdot 0 = 0$. $-\log \sigma(0) = -\log(0.5) = \log 2 = 0.6931$. Este es **el punto de partida de DPO**: cuando la policy y el ref son iguales, la loss es exactamente `log 2`. A medida que entrenamos, la policy se desviara del ref para FAVORECER chosen sobre rejected, los ratios se separaran, y la loss bajara. Si en cap 29 ves la loss empezar en 0.69 y descender — todo va bien. Si empieza en 0.30 — algo esta mal cargado. Si nunca baja de 0.69 — el dataset no tiene señal o `beta` es 0.

**El helper coincide.** El assert pasa: nuestro `dpo_loss` en `_models.py` reproduce el calculo manual con error <1e-4. La derivacion es correcta y la implementacion es correcta. Listo para entrenar (cap 28-29).

---

## 7. Que esperamos durante DPO training

Cuando entrenamos en cap 29, esperamos:

- `log pi_theta(y_w|x)` SUBE (vs ref): la policy le da mas probabilidad al chosen.
- `log pi_theta(y_l|x)` BAJA (vs ref): la policy le da menos probabilidad al rejected.
- `ratio_w > ratio_l` $\Rightarrow z > 0 \Rightarrow$ loss < 0.6931.
- En el limite, `ratio_w >> ratio_l` $\Rightarrow z \gg 0 \Rightarrow$ loss $\to 0$.

Pero el `beta` y el KL implicito limitan cuanto puede subir el ratio_w. La policy no puede ignorar al ref completamente. Es la diferencia central con SFT: SFT empuja la prob del target sin restriccion (puede destruir cualquier conocimiento previo); DPO empuja la prob del chosen pero anclando al SFT como ancla. Por eso DPO suele dejar el modelo mas robusto que un SFT muy largo.

---

## 8. Preguntas de verificacion

1. ¿Por que `pi_ref` debe estar congelado? — Si el ref tambien se moviera, el KL implicito desapareceria y la policy podria colapsar a cualquier distribucion sin penalizacion. El ref es el ancla: solo tiene sentido medir "cuanto te has desviado del SFT" si el SFT no se mueve.

2. Si `beta = 0`, ¿que pasa? — $z = 0$ siempre, la loss queda constante en $\log 2$, y el gradiente respecto a los parametros es 0. El modelo no aprende nada. `beta` controla la magnitud de la señal, no solo el equilibrio KL.

3. Para un dataset donde el chosen y rejected estan inversamente correlacionados con la preferencia real, ¿que pasaria? — La loss bajaria igual (mecanicamente sigue minimizando $-\log \sigma(z)$ con `z` definido por las etiquetas), pero el modelo aprenderia a EVITAR el chosen "real" y a preferir el rejected "real". DPO confia ciegamente en las etiquetas de preferencia. La calidad del dataset es todo.
