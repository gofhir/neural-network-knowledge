---
title: "Direct Preference Optimization (DPO)"
weight: 289
math: true
---

**Direct Preference Optimization (DPO)** es la tecnica introducida por **Rafailov et al. (NeurIPS 2023)** que reemplaza el pipeline clasico de RLHF -- entrenar un reward model y despues hacer PPO -- por **una sola loss supervisada** sobre pares (chosen, rejected) que opera directamente sobre la policy. Su valor practico es enorme: elimina dos de las tres fases de RLHF (reward model y PPO), no requiere sampling on-policy, y empiricamente alcanza calidad comparable a PPO-RLHF con una fraccion del compute y la mitad del dolor de cabeza de ingenieria.

DPO se ha vuelto el default de alineacion offline en 2024-2026: aparece en la receta de Llama-3 Instruct, Mistral Instruct, Zephyr, Tulu, OLMo Instruct, y la mayoria de modelos open-weight. Es el corazon del *Camino 2* del curso.

---

## 1. Apertura: Rafailov 2023, evitar reward model + PPO

Antes de DPO, alinear un LLM con preferencias humanas era una receta de tres fases (Christiano 2017, InstructGPT 2022):

1. **SFT**: ajustar el base model con pares (instruccion, respuesta).
2. **Reward model (RM)**: entrenar un MLP $r_\phi(x, y)$ con loss [Bradley-Terry](/fundamentos/bradley-terry) sobre preferencias humanas.
3. **PPO**: optimizar la policy $\pi_\theta$ para maximizar $r_\phi$ con KL penalty hacia $\pi_{\text{SFT}}$.

Cada fase tiene sus dolores: el RM puede sufrir reward hacking, PPO requiere sampling on-policy caro y es notoriamente sensible a hyperparams (clip epsilon, GAE lambda, reward normalization, advantage clipping). En la practica InstructGPT documenta que entrenar PPO requiere mas tuning que SFT y RM combinados.

Rafailov et al. demostraron que **toda la fase RM+PPO se puede colapsar en una sola loss**, derivable analiticamente desde el mismo objetivo regularizado que PPO-RLHF resuelve. La derivacion es elegante; el resultado es operacionalmente simple.

---

## 2. La idea matematica: forma cerrada de la policy optima

PPO-RLHF resuelve:

$$
\max_\pi \; \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta \, D_{\text{KL}}(\pi \| \pi_{\text{ref}}).
$$

Se puede demostrar (calculo variacional / Lagrange) que la policy optima tiene forma cerrada:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right).
$$

Es decir, $\pi^*$ es $\pi_{\text{ref}}$ **inclinada** (tilted) por el reward, normalizada por $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)$.

Despejando $r$:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

**Insight central**: $r$ se puede escribir en terminos de $\pi^*$ y $\pi_{\text{ref}}$. No necesitamos parametrizar $r$ aparte si parametrizamos $\pi^*$.

---

## 3. La loss DPO completa

Sustituyendo $r$ en la loss [Bradley-Terry](/fundamentos/bradley-terry) $-\mathbb{E}\log\sigma(r_w - r_l)$, los terminos $\beta \log Z(x)$ **se cancelan** porque dependen solo de $x$ y aparecen restandose:

$$
\boxed{
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
}
$$

Componentes:

- $\pi_\theta$: la **policy** que entrenamos (parametros aprendibles).
- $\pi_{\text{ref}}$: la **reference**, congelada. En la practica es el modelo SFT del cual partimos.
- $\beta$: hiperparametro que controla el [KL implicito](/fundamentos/kl-implicito).
- $(x, y_w, y_l)$: triple del dataset DPO -- prompt, chosen, rejected.

La loss se calcula computando 4 log-probabilidades por triple: $\log \pi_\theta(y_w \mid x)$, $\log \pi_\theta(y_l \mid x)$, $\log \pi_{\text{ref}}(y_w \mid x)$, $\log \pi_{\text{ref}}(y_l \mid x)$. Las dos de $\pi_{\text{ref}}$ no requieren gradiente (se calculan con `torch.no_grad()`).

---

## 4. Policy vs reference: por que dos modelos

DPO carga **dos copias** del modelo en memoria:

- **Policy** $\pi_\theta$: trainable, recibe gradientes, se actualiza cada step.
- **Reference** $\pi_{\text{ref}}$: frozen (`requires_grad=False`), solo se usa en forward pass.

Ambos arrancan con los mismos pesos (los del SFT). La policy se aleja conforme entrena; la reference no se mueve. La diferencia entre las dos es lo que codifica "la mejora aprendida via preferencias".

¿Por que necesitamos la reference? Por dos razones complementarias:

1. **Anclaje**: el log-ratio $\log \pi_\theta / \pi_{\text{ref}}$ codifica la divergencia de la policy respecto a un baseline conocido. Sin reference, no hay forma de penalizar derivas neutrales (cosas que cambian sin razon).
2. **Cancelacion de $Z(x)$**: el truco matematico que hace la loss tractable depende exactamente de tener una distribucion de comparacion. Sin reference, $Z(x)$ no se cancela y la loss se vuelve intractable.

En practica, hay variantes que **dropean la reference** (por ejemplo CPO, ORPO con reference-free objective) usando aproximaciones, pero el DPO clasico siempre necesita las dos.

Costo en memoria: si $\pi_\theta$ pesa N GB, la reference agrega ~N GB. Para Llama-3-70B, esto es relevante. Tecnicas: cargar reference en CPU y mover por step, usar QLoRA en la policy, o pre-computar $\log \pi_{\text{ref}}$ una vez y guardar en disco (caching).

---

## 5. Beta: el hiperparametro KL

$\beta$ aparece multiplicando ambos log-ratios. Su rol es exactamente el del coeficiente $\lambda$ del KL en PPO-RLHF, pero **invertido en signo** -- valores grandes de $\beta$ corresponden a regularizacion **fuerte** (la policy se queda cerca de la reference).

Tres regimenes:

- **$\beta$ pequeno ($\sim 0.01$)**: tolerancia alta a divergencia. La policy puede alejarse mucho de la reference. Riesgo: deriva semantica, alucinaciones nuevas.
- **$\beta$ moderado ($\sim 0.1$)**: el default de Llama-Chat, Zephyr, Mistral Instruct. Balance entre adherencia y aprendizaje.
- **$\beta$ grande ($\sim 0.5$ - $1.0$)**: regularizacion fuerte, la policy apenas se mueve. Util si la reference ya es muy buena y solo queremos correcciones quirurgicas.

Tuning practico: empieza en $\beta = 0.1$, evalua chosen-rejected reward gap. Si el gap no crece, baja $\beta$. Si crece pero la calidad subjetiva empeora, sube $\beta$.

Ver [KL implicito](/fundamentos/kl-implicito) para una discusion detallada del rol de $\beta$.

---

## 6. Por que funciona en lugar de PPO

Ventajas de DPO sobre PPO-RLHF:

- **Una fase en vez de dos**: solo DPO sobre el SFT, no RM + PPO.
- **Sin sampling on-policy**: dataset offline, fixed. PPO necesita generar respuestas en cada step.
- **Sin reward model separado**: ahorras los parametros y el compute de entrenarlo.
- **Mas estable**: menos hyperparams criticos. PPO tiene 8+ hyperparams sensibles (clip range, GAE lambda, n epochs, value loss coef, advantage normalization). DPO tiene 2 ($\beta$ y lr).
- **Compute**: 5-10x menos GPU-hours que PPO equivalente.

Limitaciones:

- **Offline**: no puede explorar respuestas que no esten en el dataset. Si $\pi_\theta$ va a regiones del espacio de salida no cubiertas por las preferencias, no aprende.
- **Sensible a la calidad del dataset**: si las preferencias son ruidosas o mal etiquetadas, DPO las absorbe sin filtros (PPO con un buen RM puede regularizar).
- **Reward gap creciente** sin garantia de calidad: la metrica que DPO optimiza (chosen-rejected log-ratio) puede crecer mientras la calidad subjetiva no mejora -- en casos extremos, la policy aprende a hacer chosen y rejected ambas peores, manteniendo el gap.
- **Memoria 2x** por la reference congelada.

Empiricamente, en benchmarks como MT-Bench, AlpacaEval, Arena Hard, DPO alcanza dentro de ~1-2% de la calidad de PPO-RLHF a una fraccion del costo. Para alineacion offline con dataset razonable, es la opcion dominante.

---

## 7. Datasets DPO

Tres familias practicas:

- **Anthropic HH-RLHF**: ~170k pares de Helpful + Harmless. Anotaciones humanas. Estandar de referencia para experimentos academicos.
- **UltraFeedback / OpenAssistant**: pares construidos rankeando salidas de varios LLMs y filtrando por GPT-4. Calidad-costo razonable.
- **Sintetico base-sampled**: generas $K$ respuestas con $\pi_{\text{SFT}}$ a temperatura alta, las rankeas con un modelo profesor (GPT-4, Claude), tomas la mejor como `chosen` y la peor como `rejected`. Es la receta moderna mas barata; aparece en Tulu, Zephyr, OpenChat.

Formato tipico:

```json
{
  "prompt":   "INSTR: traduce 'hello' al espanol\nRESP: ",
  "chosen":   "hola",
  "rejected": "Hello (no se traducir)"
}
```

Tamanos tipicos: 10k-200k triples. Mas alla de ~500k, el rendimiento marginal es bajo y aparece overfitting.

---

## 8. Codigo PyTorch

Helper directo del modulo `_models.py` del curso (cap 27 / cap 29):

```python
import torch
import torch.nn.functional as F

def compute_logp_response(model, prompt_ids, response_ids, device=None):
    """log P(response | prompt) = sum log p_t para tokens de response."""
    if device is None:
        device = next(model.parameters()).device
    full = torch.cat([prompt_ids, response_ids]).to(device).unsqueeze(0)
    inp = full[:, :-1]
    tgt = full[:, 1:]
    logits, _ = model(inp)                         # (1, T-1, V)
    logp = torch.log_softmax(logits, dim=-1)
    n_p = prompt_ids.shape[0]
    resp_logits = logp[:, n_p-1:, :]               # (1, R, V)
    resp_targets = tgt[:, n_p-1:].unsqueeze(-1)    # (1, R, 1)
    chosen = resp_logits.gather(-1, resp_targets).squeeze(-1)
    return chosen.sum()


def dpo_loss(policy, ref, prompt_ids, chosen_ids, rejected_ids, beta=0.1):
    """DPO loss para un triple (x, y_w, y_l)."""
    logp_w_pi = compute_logp_response(policy, prompt_ids, chosen_ids)
    logp_l_pi = compute_logp_response(policy, prompt_ids, rejected_ids)
    with torch.no_grad():
        logp_w_ref = compute_logp_response(ref, prompt_ids, chosen_ids)
        logp_l_ref = compute_logp_response(ref, prompt_ids, rejected_ids)
    log_ratio_w = logp_w_pi - logp_w_ref
    log_ratio_l = logp_l_pi - logp_l_ref
    return -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
```

Tres lecciones del codigo:

1. **`with torch.no_grad()` en la reference**: ahorra memoria y compute, no necesitamos gradientes.
2. **Sumatoria sobre todos los tokens de response**: $\log \pi(y \mid x) = \sum_t \log p_t$. Algunas variantes (length-normalized DPO) dividen por el numero de tokens.
3. **`F.logsigmoid` en vez de `torch.log(torch.sigmoid(...))`**: estabilidad numerica, evita underflow para valores grandes negativos.

Para batches, vectorizar las cuatro llamadas a `compute_logp_response` y promediar al final.

---

## 9. Resumen

- **DPO** elimina la fase RM + PPO de RLHF: una sola loss sobre pares (chosen, rejected) que opera sobre la policy.
- Se deriva sustituyendo la **forma cerrada** de la policy optima (de PPO-KL) en la loss [Bradley-Terry](/fundamentos/bradley-terry).
- La loss tiene la forma $-\log\sigma(\beta(\log\pi_\theta/\pi_{\text{ref}})_w - \beta(\log\pi_\theta/\pi_{\text{ref}})_l)$.
- Requiere **dos copias del modelo** (policy trainable, reference congelada) -- ambas arrancan en el SFT.
- $\beta$ controla el [KL implicito](/fundamentos/kl-implicito); default $\sim 0.1$.
- **Ventajas**: una fase, sin sampling on-policy, sin RM, mas estable, 5-10x mas barato que PPO.
- **Limitaciones**: offline puro, sensible a calidad de preferencias, reward gap puede crecer sin mejora subjetiva.
- **Default de 2024-2026** para alineacion offline; aparece en Llama-3, Mistral, Zephyr, Tulu, OLMo.

## Ver tambien

- [Bradley-Terry](/fundamentos/bradley-terry) -- la base de la loss DPO.
- [SFT](/fundamentos/sft) -- el paso previo que produce $\pi_{\text{ref}}$.
- [KL Implicito](/fundamentos/kl-implicito) -- el regularizador que aparece dentro de la sigmoide.
- [Loss Masking](/fundamentos/loss-masking) -- relevante para calcular `compute_logp_response` correctamente sobre tokens de respuesta.
- [Foundation Models](/fundamentos/foundation-models) -- contexto del paradigma pretrain + adapt.
- [Clase 14 cap 27 - DPO loss](/clases/clase-14/practica/27-dpo-loss) -- la derivacion completa.
- [Clase 14 cap 28 - Dataset DPO](/clases/clase-14/practica/28-dataset-dpo) -- construccion del dataset.
