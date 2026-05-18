---
title: "05 - RLHF toy pipeline"
weight: 35
math: true
---

## Encuadre

Hasta acá hemos visto RLHF como concepto (Capítulo de teoría de la clase 20) y como pieza histórica clave en InstructGPT y ChatGPT (ver [paper InstructGPT](/papers/instructgpt-ouyang-2022) y [fundamento RLHF](/fundamentos/rlhf)). En este capítulo bajamos a fierros: implementamos el pipeline completo en tres pasos sobre un modelo realmente pequeño, GPT-2 base (124M parámetros), usando TRL (Transformer Reinforcement Learning) de HuggingFace.

El objetivo no es lograr un asistente útil. El objetivo es que cada paso del pipeline sea ejecutable, observable y dejen evidencia empírica de los modos de falla que la literatura describe en escala: reward hacking, mode collapse, alignment tax. Si ya viste esos fenómenos en juguete, los vas a reconocer cuando aparezcan en producción.

Los tres pasos clásicos son:

1. **SFT** (Supervised Fine-Tuning): adaptar el modelo base a seguir instrucciones simples vía clonación de comportamiento.
2. **Reward Model**: entrenar un modelo separado que aprende a puntuar respuestas según preferencias humanas (o, en nuestro caso, un proxy sintético).
3. **PPO**: optimizar la policy (el modelo SFT) contra el reward model, regularizada por una penalización KL hacia la policy SFT original.

Toda la corrida cabe en una sola GPU consumer (8-12 GB VRAM bastan) o incluso CPU si bajas tamaños. La meta es que termine en menos de una hora en una T4 de Colab gratuita.

Cross-links:

- [Paper InstructGPT](/papers/instructgpt-ouyang-2022) — la receta original que popularizó RLHF
- [Fundamento RLHF](/fundamentos/rlhf) — matemática y motivación
- [Fundamento SFT](/fundamentos/sft) — paso 1 en detalle
- [Fundamento DPO](/fundamentos/dpo) — la alternativa más simple que veremos al final

Nota importante: la API de `trl` cambia con frecuencia. Este capítulo está anclado a **TRL >= 0.11**. Si usas una versión anterior verás que `SFTTrainer`, `RewardTrainer` y `PPOTrainer` tenían firmas distintas (por ejemplo `tokenizer=` en vez de `processing_class=`, o `PPOTrainer.step()` manual en lugar de `trainer.train()`). Revisa el changelog de TRL antes de copiar cualquier código si tu versión instalada difiere. El equipo de HuggingFace ha hecho varios pases de cleanup de API entre 0.7 y 0.11, lo cual rompió código que estaba en blogs y notebooks circulando online.

### Por qué hacer el toy antes que el real

Tres razones concretas para no saltarse este ejercicio aun cuando uno ya entendió la teoría:

1. **Las curvas son legibles**. En 124M params con dataset sintético, las series de wandb (reward, KL, entropy) se ven limpias y se puede asociar cada pico a una causa identificable. En 7B+ con dataset humano real, las curvas son ruidosas y los efectos están entrelazados.
2. **Los modos de fallo son reproducibles**. Reward hacking en escala te lleva semanas detectar. En toy lo ves en 100 steps. Es un terreno donde se puede iterar sobre la intuición.
3. **El stack mental queda armado**. Cuando arme un pipeline real con tu propio dataset de preferencias (médico, legal, lo que sea), no vas a estar peleando con la API de TRL al mismo tiempo que con la calidad de las anotaciones. El skill RLHF y el skill domain-data se desacoplan.

## Setup

```bash
pip install "transformers>=4.45" "datasets>=2.20" "trl>=0.11" peft accelerate bitsandbytes wandb
```

Imports y semillas reproducibles:

```python
import os
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"  # 124M parametros

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 no trae pad token por defecto; necesario para batching
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

Wandb es opcional pero recomendado para ver curvas en vivo:

```python
import wandb
wandb.init(project="rlhf-toy-clase20", mode="online")  # mode="disabled" si no quieres tracking
```

## Paso 1: SFT (Supervised Fine-Tuning)

### Por qué SFT primero

Un modelo de lenguaje preentrenado en corpus web no sabe que "responder a una pregunta" es algo deseable. Si le das "¿Cuál es la capital de Francia?" sin contexto, GPT-2 base lo más probable es que continúe con más preguntas, con un titular periodístico o con una digresión, porque eso es lo que vio en la web. SFT le enseña la convención de "pregunta → respuesta corta y directa" por imitación.

Formalmente, SFT minimiza la log-verosimilitud negativa de la respuesta condicionada al prompt:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})\right]$$

Es exactamente el mismo objetivo que el preentrenamiento, pero sobre un dataset curado de demostraciones (x, y).

### Dataset sintético de pares pregunta-respuesta

Para mantener todo trazable usamos un dataset sintético de 200 pares "¿Cuál es la capital de X?" donde X cubre países latinoamericanos, europeos y asiáticos. La respuesta target tiene forma fija: "La capital de X es Y."

```python
CAPITALES = {
    "Chile": "Santiago", "Argentina": "Buenos Aires", "Peru": "Lima",
    "Brasil": "Brasilia", "Mexico": "Ciudad de Mexico", "Colombia": "Bogota",
    "Uruguay": "Montevideo", "Paraguay": "Asuncion", "Bolivia": "La Paz",
    "Ecuador": "Quito", "Venezuela": "Caracas", "Espana": "Madrid",
    "Francia": "Paris", "Italia": "Roma", "Alemania": "Berlin",
    "Portugal": "Lisboa", "Inglaterra": "Londres", "Japon": "Tokio",
    "China": "Beijing", "India": "Nueva Delhi",
}

def build_sft_dataset(n_per_pais: int = 10) -> Dataset:
    rows = []
    for pais, capital in CAPITALES.items():
        for _ in range(n_per_pais):
            prompt = f"Pregunta: Cual es la capital de {pais}?\nRespuesta:"
            answer = f" La capital de {pais} es {capital}."
            rows.append({"text": prompt + answer})
    random.shuffle(rows)
    return Dataset.from_list(rows)

sft_dataset = build_sft_dataset(n_per_pais=10)
print(sft_dataset[0])
# {'text': 'Pregunta: Cual es la capital de Chile?\nRespuesta: La capital de Chile es Santiago.'}
```

200 ejemplos es muy poco para fine-tunear de verdad, pero para que GPT-2 aprenda esta convención superficial (que el token después de "Respuesta:" debe ser una oración declarativa corta) sobra.

### Entrenamiento con SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft-toy",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    max_length=128,
    report_to="wandb",
    seed=SEED,
)

trainer = SFTTrainer(
    model=MODEL_NAME,
    args=sft_config,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("./sft-toy/final")
```

Detalles del SFTTrainer en TRL >= 0.11:

- `processing_class` reemplaza al antiguo `tokenizer=` (warning de deprecación si usas el viejo).
- Detecta automáticamente la columna `text` y la tokeniza.
- Aplica completion-only loss si pasas `formatting_func` con separador, pero acá entrenamos sobre todo el texto para simplicidad.

### Eval cualitativa

```python
from transformers import pipeline

def generate(model_path: str, prompts: list[str], **gen_kwargs) -> list[str]:
    pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)
    outs = pipe(prompts, max_new_tokens=30, do_sample=False, **gen_kwargs)
    return [o[0]["generated_text"] for o in outs]

test_prompts = [
    "Pregunta: Cual es la capital de Chile?\nRespuesta:",
    "Pregunta: Cual es la capital de Francia?\nRespuesta:",
    "Pregunta: Cual es la capital de Japon?\nRespuesta:",
]

print("=== Base GPT-2 ===")
for o in generate("gpt2", test_prompts):
    print(o); print("---")

print("=== SFT ===")
for o in generate("./sft-toy/final", test_prompts):
    print(o); print("---")
```

Salida típica (abreviada):

```text
=== Base GPT-2 ===
Pregunta: Cual es la capital de Chile?
Respuesta: A: ¿Y la economia? Pregunta: Cual es...
---
=== SFT ===
Pregunta: Cual es la capital de Chile?
Respuesta: La capital de Chile es Santiago.
---
```

El base diverge en preguntas encadenadas o ruido. El SFT respondió en formato canónico. Esto es lo único que SFT te garantiza: aprender la forma. No te garantiza que la respuesta sea correcta sobre países que no vio (vamos a probar eso después con RLHF).

## Paso 2: Reward Model

### Motivación

SFT te da "responde en formato pregunta-respuesta". No te dice cuál respuesta es mejor entre dos opciones plausibles. Si el modelo SFT a temperatura 0.8 genera dos completions distintos, ¿cuál preferimos?

En el flujo real de InstructGPT, contratan labelers humanos para que ranqueen pares. En toy reemplazamos al labeler humano por un proxy reward determinístico que codifica una preferencia simple: respuestas cortas y directas son mejores que respuestas largas con divagaciones.

### Generación de dataset de preferencias

```python
import torch
from transformers import AutoModelForCausalLM

sft_model = AutoModelForCausalLM.from_pretrained("./sft-toy/final").to(DEVICE)
sft_model.eval()

@torch.no_grad()
def sample_two(prompt: str, temp_a: float = 0.3, temp_b: float = 1.2) -> tuple[str, str]:
    """Genera dos completions con temperaturas distintas. Temp baja tiende a respuestas mas directas."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out_a = sft_model.generate(
        **inputs, max_new_tokens=30, do_sample=True, temperature=temp_a,
        pad_token_id=tokenizer.eos_token_id,
    )
    out_b = sft_model.generate(
        **inputs, max_new_tokens=30, do_sample=True, temperature=temp_b,
        pad_token_id=tokenizer.eos_token_id,
    )
    text_a = tokenizer.decode(out_a[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    text_b = tokenizer.decode(out_b[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text_a, text_b

def proxy_reward(text: str) -> float:
    """Reward sintetico que prefiere:
       - respuestas cortas (penaliza largo)
       - sin repeticion de tokens
       - que empiecen con 'La capital'
    """
    tokens = text.strip().split()
    if not tokens:
        return -10.0
    length_penalty = -0.1 * max(0, len(tokens) - 8)  # ideal ~8 tokens
    repeat_penalty = -1.0 * (len(tokens) - len(set(tokens))) / max(1, len(tokens))
    format_bonus = 1.0 if text.strip().startswith("La capital") else 0.0
    return length_penalty + repeat_penalty + format_bonus

def make_preference_pairs(n_pairs: int = 200) -> Dataset:
    rows = []
    paises = list(CAPITALES.keys())
    while len(rows) < n_pairs:
        pais = random.choice(paises)
        prompt = f"Pregunta: Cual es la capital de {pais}?\nRespuesta:"
        text_a, text_b = sample_two(prompt)
        r_a, r_b = proxy_reward(text_a), proxy_reward(text_b)
        if abs(r_a - r_b) < 0.05:
            continue  # descartamos pares muy parecidos
        chosen, rejected = (text_a, text_b) if r_a > r_b else (text_b, text_a)
        rows.append({
            "prompt": prompt,
            "chosen": prompt + chosen,
            "rejected": prompt + rejected,
        })
    return Dataset.from_list(rows)

pref_dataset = make_preference_pairs(n_pairs=200)
print(pref_dataset[0])
```

Esto es trampa pedagógica deliberada. El "labeler" es una función. La consecuencia interesante es que en el paso 3 vamos a ver al modelo aprendiendo a explotar ese proxy (reward hacking) de formas que se ven obvias acá pero que en escala humana son mucho más difíciles de detectar.

### Entrenamiento del RM

El reward model es una arquitectura tipo `AutoModelForSequenceClassification` con `num_labels=1`: toma `(prompt, completion)` concatenados y emite un escalar.

El objetivo es el **Bradley-Terry loss** sobre pares preferidos:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\right]$$

donde $y_w$ es chosen, $y_l$ rejected, y $\sigma$ la sigmoide. La intuición: queremos que el reward asignado a la respuesta preferida sea mayor que el de la rechazada por al menos un margen, y la sigmoide convierte la diferencia en probabilidad de "ganar".

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=1
)
reward_model.config.pad_token_id = tokenizer.pad_token_id

reward_config = RewardConfig(
    output_dir="./rm-toy",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=1e-5,
    logging_steps=10,
    max_length=128,
    report_to="wandb",
    seed=SEED,
)

reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)
reward_trainer.train()
reward_trainer.save_model("./rm-toy/final")
```

Detalles importantes:

- Inicializamos desde `gpt2` (no desde el SFT) por simplicidad. En InstructGPT inicializan el RM desde el SFT model y reemplazan la cabeza, lo cual es marginalmente mejor pero más código. Para toy basta.
- `num_train_epochs=2` es deliberadamente bajo. El RM se sobreajusta rápido a datasets pequeños, lo cual perjudica al PPO posterior porque el reward se vuelve degenerado fuera del soporte de entrenamiento.

### Eval del RM

```python
import torch.nn.functional as F

@torch.no_grad()
def score(text: str, rm) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    return rm(**inputs).logits[0, 0].item()

rm = AutoModelForSequenceClassification.from_pretrained("./rm-toy/final").to(DEVICE)
rm.eval()

correct = 0
for ex in pref_dataset.select(range(50)):
    s_chosen = score(ex["chosen"], rm)
    s_rejected = score(ex["rejected"], rm)
    if s_chosen > s_rejected:
        correct += 1
print(f"RM accuracy en train: {correct/50:.2%}")
```

Esperamos > 80% de accuracy. Si no lo logras, sube `num_train_epochs` o revisa que las longitudes sean consistentes (un RM puede aprender a usar el largo como atajo, lo cual es exactamente el reward hacking que queremos observar después).

## Paso 3: PPO con KL penalty

### Objetivo combinado

PPO optimiza la policy $\pi_\theta$ inicializada desde el SFT model, tratando la generación de tokens como un MDP donde la recompensa total es la suma de:

1. El reward del RM evaluado sobre el completion completo: $r_\phi(x, y)$.
2. Una penalización KL token-a-token contra la policy de referencia (el SFT congelado):

$$R(x, y) = r_\phi(x, y) - \beta \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{SFT}}(y_t \mid x, y_{<t})}$$

El término KL es crítico. Sin él, la policy puede colapsar en una secuencia degenerada que maximice el RM (típicamente repitiendo la frase que el RM puntúa alto). El coeficiente $\beta$ (en TRL: `kl_coef`) controla cuánto le permitimos al modelo alejarse del SFT.

La actualización de policy usa el clipped surrogate objective de PPO:

$$\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}\left[\min\big(\rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\big)\right]$$

donde $\rho_t(\theta) = \pi_\theta(y_t) / \pi_{\theta_{\text{old}}}(y_t)$ es el ratio de probabilidad y $\hat{A}_t$ es la ventaja estimada (con GAE típicamente). El clipping evita actualizaciones catastróficas que alejen demasiado a $\pi_\theta$ de $\pi_{\theta_{\text{old}}}$ en una sola iteración.

### Setup PPO en TRL

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM

# Policy: se inicializa desde SFT y va a entrenarse
policy = AutoModelForCausalLM.from_pretrained("./sft-toy/final").to(DEVICE)

# Reference policy: copia congelada del SFT, para la KL
ref_policy = AutoModelForCausalLM.from_pretrained("./sft-toy/final").to(DEVICE)
for p in ref_policy.parameters():
    p.requires_grad = False

# Reward model entrenado en paso 2
reward_model_ppo = AutoModelForSequenceClassification.from_pretrained("./rm-toy/final").to(DEVICE)
for p in reward_model_ppo.parameters():
    p.requires_grad = False

# Value model: PPO necesita un critic. En TRL se puede compartir backbone con el RM o usar otro
value_model = AutoModelForSequenceClassification.from_pretrained("./rm-toy/final").to(DEVICE)

# Dataset de solo prompts (sin respuestas; las genera la policy en cada step)
def build_prompt_dataset(n: int = 500) -> Dataset:
    rows = []
    paises = list(CAPITALES.keys())
    for _ in range(n):
        pais = random.choice(paises)
        rows.append({"input_ids": tokenizer(
            f"Pregunta: Cual es la capital de {pais}?\nRespuesta:",
            return_tensors="pt"
        ).input_ids[0]})
    return Dataset.from_list(rows)

prompt_dataset = build_prompt_dataset(500)

ppo_config = PPOConfig(
    output_dir="./ppo-toy",
    per_device_train_batch_size=2,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_ppo_epochs=4,
    kl_coef=0.02,
    num_train_epochs=1,
    response_length=30,
    temperature=0.9,
    report_to="wandb",
    seed=SEED,
)

ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model_ppo,
    value_model=value_model,
    train_dataset=prompt_dataset,
    processing_class=tokenizer,
)
ppo_trainer.train()
ppo_trainer.save_model("./ppo-toy/final")
```

### Qué observar durante el entrenamiento

Wandb (o los logs en consola) te muestra varias series temporales clave:

| Métrica                 | Qué significa                          | Qué esperar                                          |
| ----------------------- | -------------------------------------- | ---------------------------------------------------- |
| `objective/scores`      | Reward promedio del RM por batch       | Debe subir                                           |
| `objective/kl`          | KL acumulada de policy vs ref          | Sube despacio; explotar = mode collapse              |
| `objective/rlhf_reward` | Reward total (RM − β·KL)               | Es el que PPO optimiza                               |
| `policy/clipfrac`       | Fracción de updates clippeadas         | Idealmente entre 0.1-0.3                             |
| `policy/entropy`        | Entropía de la policy                  | Baja con entrenamiento; si cae a cero, mode collapse |

Si ves `objective/kl` disparándose mientras `policy/entropy` se va al suelo, estás en mode collapse: el modelo encontró una secuencia que el RM ama y la repite mecánicamente. Subir `kl_coef` o bajar `learning_rate` lo mitiga.

### Eval final: comparación base / SFT / RLHF

```python
test_prompts_eval = [
    "Pregunta: Cual es la capital de Chile?\nRespuesta:",
    "Pregunta: Cual es la capital de Francia?\nRespuesta:",
    "Pregunta: Cual es la capital de Japon?\nRespuesta:",
    "Pregunta: Cual es la capital de Noruega?\nRespuesta:",  # OOD: no esta en CAPITALES
    "Pregunta: Cual es la capital de Marte?\nRespuesta:",    # adversarial
]

variants = {
    "base":  "gpt2",
    "sft":   "./sft-toy/final",
    "rlhf":  "./ppo-toy/final",
}

for name, path in variants.items():
    print(f"\n=== {name.upper()} ===")
    for o in generate(path, test_prompts_eval):
        print(o.replace("\n", " | "))
```

Salida ilustrativa (los números varían por seed):

```text
=== BASE ===
... Cual es la capital de Chile? | Respuesta: A: La capital es la de la...
... Cual es la capital de Noruega? | Respuesta: Pregunta 2: ¿Que paises...
=== SFT ===
... Cual es la capital de Chile? | Respuesta: La capital de Chile es Santiago.
... Cual es la capital de Noruega? | Respuesta: La capital de Noruega es Estocolmo.  # incorrecto pero formato OK
=== RLHF ===
... Cual es la capital de Chile? | Respuesta: La capital de Chile es Santiago.
... Cual es la capital de Noruega? | Respuesta: La capital de Noruega es Oslo.
... Cual es la capital de Marte? | Respuesta: La capital de Marte es La capital.  # reward hacking visible
```

Nota: RLHF puede ser correcto donde SFT alucinaba (si los pares de preferencia ayudaron al modelo a calibrar mejor), pero también puede degenerar en prompts adversariales que no estaban en la distribución del RM.

## Alternativa más simple: DPO

DPO (Direct Preference Optimization, Rafailov et al. 2023) es la principal razón por la que en 2024-2026 mucha gente ya no implementa PPO en serio. DPO **salta el paso 2 y el paso 3**: optimiza la policy directamente sobre el dataset de preferencias, sin reward model ni rollouts. La derivación clave es que el óptimo de RLHF tiene una forma cerrada en términos del ratio de log-prob de policy vs reference, y ese ratio puede entrenarse con un loss tipo classification:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

En TRL:

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo-toy",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=5e-6,
    beta=0.1,
    max_length=128,
    max_prompt_length=64,
    logging_steps=10,
    report_to="wandb",
    seed=SEED,
)

dpo_model = AutoModelForCausalLM.from_pretrained("./sft-toy/final").to(DEVICE)
dpo_ref = AutoModelForCausalLM.from_pretrained("./sft-toy/final").to(DEVICE)

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=dpo_ref,
    args=dpo_config,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model("./dpo-toy/final")
```

Cuatro líneas conceptuales y se acabó. Sin RM. Sin rollouts. Sin GAE. Sin clipped ratios. Sin debugging de PPO.

### Cuándo PPO vs DPO

| Criterio                  | PPO                                     | DPO                                    |
| ------------------------- | --------------------------------------- | -------------------------------------- |
| Complejidad de pipeline   | Alta (3 modelos en RAM)                 | Baja (2 modelos)                       |
| Estabilidad               | Sensible a hyperparams                  | Más estable                            |
| Reward model reutilizable | Sí, sirve para inference-time best-of-N | No hay RM explícito                    |
| Online RL (exploración)   | Sí (genera y aprende)                   | No (offline sobre preferencias fijas)  |
| Sample efficiency         | Menor (necesita rollouts)               | Mayor                                  |
| Estado del arte 2024+     | Aún usado en frontier labs              | Default para fine-tunes abiertos       |

La regla práctica: si tienes preferencias estáticas y solo querés alinear, DPO. Si querés iterar con el modelo en el loop (RLAIF, online RLHF, RLVR), PPO o sus variantes (GRPO, RLOO). Ver [fundamento DPO](/fundamentos/dpo) para la derivación completa y discusión de variantes (IPO, KTO, ORPO).

## Análisis y comparación final

Métricas que vale la pena reportar sobre los 4 modelos sobre un set de 20 prompts test:

```python
@torch.no_grad()
def evaluate_model(path: str, rm, prompts: list[str]) -> dict:
    pipe = pipeline("text-generation", model=path, tokenizer=tokenizer,
                    device=0 if DEVICE=="cuda" else -1)
    outs = pipe(prompts, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
    completions = [o[0]["generated_text"] for o in outs]
    lengths = [len(c.split()) for c in completions]
    rewards = [score(c, rm) for c in completions]
    return {
        "avg_length": np.mean(lengths),
        "avg_reward": np.mean(rewards),
        "samples": completions[:3],
    }

results = {name: evaluate_model(path, rm, test_prompts_eval) for name, path in {
    "base": "gpt2",
    "sft":  "./sft-toy/final",
    "ppo":  "./ppo-toy/final",
    "dpo":  "./dpo-toy/final",
}.items()}
```

Tabla resumen ilustrativa:

| Modelo     | Avg length | Avg RM reward | Formato OK |
| ---------- | ---------- | ------------- | ---------- |
| base GPT-2 | 28 tokens  | -1.8          | 0 / 5      |
| SFT        | 9 tokens   | +0.4          | 5 / 5      |
| PPO (RLHF) | 7 tokens   | +1.2          | 5 / 5      |
| DPO        | 8 tokens   | +1.0          | 5 / 5      |

Lecturas:

- SFT ya hace el grueso del trabajo en este toy (lleva el formato de 0 a 5/5).
- PPO y DPO empujan el reward del RM hacia arriba, lo cual era el objetivo del entrenamiento.
- Que el reward suba **no significa que el modelo sea mejor**. Significa que aprendió a complacer al RM, que es un proxy imperfecto.

## Modos de fallo observables incluso en toy

### Reward hacking

Nuestro `proxy_reward` premia respuestas que empiecen con "La capital". Si entrenas PPO suficientes steps con `kl_coef` bajo, vas a ver outputs como:

```text
La capital de Marte es La capital.
La capital de cualquier La capital La capital La capital.
```

El modelo descubrió que repetir el prefix "La capital" maximiza el reward del RM (porque el RM aprendió que ese token al inicio correlaciona con respuestas chosen). El RM no penaliza repetición tanto como debería, y la policy explota ese hueco.

Este patrón es la versión de juguete del fenómeno descrito por Skalse et al. 2022 ("Defining and Characterizing Reward Hacking") y por Gao, Schulman & Hilton 2023 ("Scaling Laws for Reward Model Overoptimization"). El último paper en particular muestra una ley empírica: el reward del RM sigue subiendo durante PPO mientras la calidad real (medida por humanos o por un RM gold-standard mucho más grande) primero sube y luego baja, formando un arco invertido. El punto óptimo de divergencia ocurre antes de lo que uno asume; sobre-entrenar PPO contra un RM imperfecto degrada activamente al modelo aun cuando las métricas internas dicen lo contrario.

Mitigaciones reales:

- Penalización explícita en el reward por repetición/length.
- Reward shaping con múltiples señales (helpfulness + harmlessness + honestidad como en InstructGPT).
- Constitutional AI / RLAIF para diversificar fuentes de feedback.
- Early stopping basado en métricas externas al RM (judge humano o LLM-judge fuerte cada N steps).
- Ensemble de reward models y agregación conservadora (mínimo o cuantil bajo) para penalizar reward que solo un RM cree alto.

### Mode collapse

Si bajas `kl_coef` a 0.001 y entrenas 1000 steps, casi seguro vas a ver que para todos los prompts el modelo emite el mismo string. La policy encontró el argmax local del RM y la entropía colapsó. La curva `policy/entropy` cae verticalmente, `objective/kl` se va a 10+, y los outputs son idénticos.

Esto es por qué la KL penalty existe. InstructGPT usó $\beta \approx 0.02$ y en escalas mayores se necesitan a veces schedules adaptativos (KL controller). El "KL controller" más popular es el de Ziegler et al. 2019: se define un KL target $K^*$ por step y se ajusta $\beta$ multiplicativamente cuando la KL observada sale de un rango alrededor del target. Algo así:

```python
def update_kl_coef(beta_t, kl_observed, kl_target=6.0):
    proportional_error = max(-0.2, min(0.2, (kl_observed - kl_target) / kl_target))
    return beta_t * (1 + 0.1 * proportional_error)
```

En TRL >= 0.11 esto se puede activar con `adaptive_kl=True` en `PPOConfig`. Para datasets pequeños como el de este toy puede empeorar las cosas (oscila), pero en producción es estándar.

### Alignment tax

GPT-2 base, dado el prompt `"The history of artificial intelligence began"`, produce una continuación razonable. El modelo RLHF, evaluado en el mismo prompt fuera de la distribución de entrenamiento (preguntas sobre capitales), genera basura: "La capital de The es history."

Esto es alignment tax: la policy se especializó tanto en el patrón que vio durante PPO que perdió capacidades generales del base. Ouyang et al. (2022) lo reportan en escala para InstructGPT y proponen mezclar gradientes de pretrain durante PPO (`pretrain_grad_coef` en su receta, conocido también como "PPO-ptx") para mitigarlo. La idea: en cada batch de PPO, también se muestrea un batch de texto del corpus de pretrain y se calcula el loss de language modeling estándar, sumando ambos gradientes. Esto ancla a la policy a no olvidar lo que sabía. En InstructGPT esta variante recupera buena parte de la capacidad general perdida sin sacrificar la mejora en helpfulness medida por humanos.

Otra mitigación más reciente es el uso de LoRA / adapters durante RLHF: la mayoría de los pesos del SFT se congelan y solo se entrenan adapters de bajo rango. Esto limita por construcción cuánto puede alejarse la policy del SFT, funcionando como una regularización implícita. Es muy popular en pipelines RLHF caseros porque además baja drásticamente el VRAM requerido. En TRL se hace pasando `peft_config` al `PPOTrainer`.

## Limitaciones del toy pipeline

1. **Dataset sintético sin labelers humanos**: nuestro `proxy_reward` es transparente y por eso podemos auditar el reward hacking. En realidad los labelers humanos son ruidosos, inconsistentes, sesgados, y caros. Anthropic y OpenAI gastan millones en pipelines de anotación.
2. **GPT-2 124M es demasiado chico**: RLHF empieza a brillar arriba de 1B-7B params. El paper de InstructGPT muestra que el "gap" entre SFT y RLHF crece con escala. En 124M el SFT te alcanza para casi todo lo que querés en juguete.
3. **PPO es notoriamente sensible**: `kl_coef`, `learning_rate`, `cliprange`, `gamma`, `lam`, `num_ppo_epochs` interactúan no-trivialmente. Mismo seed, distinta GPU, resultados distintos. Por eso DPO ganó mindshare.
4. **Reproducibilidad limitada**: incluso con `torch.manual_seed`, la generación sampleada en el rollout de PPO depende del orden de batches, del dropout, y del determinismo de CUDA. Para papers reales hay que reportar varias seeds.
5. **No medimos preferencias humanas**: nuestra "ganancia" es contra el RM que entrenamos. Es circular. Para un test honesto haría falta humanos o un LLM-as-judge fuerte (GPT-4 class) evaluando ciegamente.
6. **No exploramos exploration vs exploitation seriamente**: el `temperature=0.9` del rollout es una elección casi arbitraria. En PPO real, el equilibrio entre explorar nuevas secuencias y explotar las que ya tienen reward alto se controla con entropy bonus, temperatura adaptativa, o muestreo top-p. Cambiarlo afecta mucho los resultados.
7. **Tokenizers de modelos modernos son distintos**: GPT-2 tiene 50k tokens BPE entrenados sobre inglés. Modelos como Llama-3 tienen vocabularios mucho mayores y mejor cobertura multilingüe. Algunos issues que veas en este toy con español acentuado (la palabra "Cual" sin tilde es una concesión deliberada al tokenizer) desaparecen automáticamente en escala.

### Cómo extender este toy hacia algo útil

Si querés convertir este pipeline en algo más cercano a producción, los pasos en orden de prioridad serían:

1. Cambiar GPT-2 por un modelo instruction-tuned pequeño (Qwen2.5-0.5B, SmolLM2-360M, TinyLlama-1.1B). Ya vienen con SFT decente y el ejercicio se concentra en RM + PPO/DPO.
2. Reemplazar el `proxy_reward` por anotaciones humanas reales (incluso 100-200 pares anotados a mano valen oro) o por LLM-as-judge con un modelo más fuerte.
3. Usar LoRA en SFT y PPO para que entrenar sea factible en hardware modesto y para limitar alignment tax.
4. Agregar un set de eval separado con prompts out-of-distribution para detectar reward hacking temprano.
5. Probar DPO antes que PPO. Si DPO ya logra lo que querés, no necesitas PPO. Mucha gente se da cuenta de esto recién después de invertir semanas en infraestructura PPO.

## Cross-links finales

- [Camino 04 — Fine-tuning con BETO](/clases/clase-20/practica/04-fine-tuning-beto) — el camino "supervisado clásico" que contrastamos con RLHF
- [Paper InstructGPT (Ouyang et al. 2022)](/papers/instructgpt-ouyang-2022) — la receta original SFT + RM + PPO
- [Fundamento RLHF](/fundamentos/rlhf) — matemática completa, GAE, KL controller adaptativo
- [Fundamento SFT](/fundamentos/sft) — el paso 1 en profundidad
- [Fundamento DPO](/fundamentos/dpo) — derivación cerrada y variantes (IPO, KTO, ORPO)
- [Profundización clase 20](/clases/clase-20/profundizacion) — discusión teórica de alineamiento más amplia

Con este capítulo cerramos el ciclo: viste RLHF como concepto (teoría), como mecanismo histórico (paper InstructGPT), y ahora como código ejecutable. La próxima vez que leas un paper de alineamiento, el vocabulario (reward hacking, KL penalty, mode collapse, alignment tax, DPO vs PPO) va a ser concreto, no abstracto.
