---
title: "El pipeline: Atari, la CNN y el experto"
weight: 2
---

DAGGER necesita tres piezas antes de arrancar: un **entorno** que produzca estados manejables, una **red** que los procese, y un **experto** consultable. Esta parte recorre esa infraestructura — que es, en gran medida, la reproducción canónica del pipeline de **DQN (Mnih et al. 2015)**.

## Preprocesamiento de Atari: los wrappers

La observación cruda de Breakout es una imagen **210×160 RGB, valores 0–255**. Pasarla directa a una red sería un desperdicio. El notebook la transforma con una cadena de *wrappers* (decoradores de entornos, adaptados de `atari_wrappers.py` de OpenAI Baselines), cada uno con una responsabilidad única:

| Wrapper | Qué hace | Por qué |
|---|---|---|
| `NoopResetEnv` | 1–30 acciones "no-op" al iniciar cada episodio | Diversifica el estado inicial → evita memorizar aperturas |
| `FireResetEnv` | Presiona FIRE al reiniciar | En Breakout hay que "servir" la pelota o el juego no arranca |
| `MaxAndSkipEnv` | Repite la acción 4 frames + máximo píxel a píxel de los 2 últimos | Frame-skip (4× más rápido) + elimina el **parpadeo** del hardware Atari |
| `WarpFrame` | RGB → gris → resize 84×84 → normalizar [0,1] | Menos datos, cuadrado para convoluciones |
| `FrameStack` | Apila los últimos 4 frames como 4 canales | Información **temporal**: de 4 fotos la red infiere dirección y velocidad |
| `EpisodicLifeEnv` | Marca `terminated` al perder **una** vida (no las 5) | Señal de aprendizaje más frecuente y nítida |

El resultado: una observación pasa de $(210,160,3)$ a $\mathbf{(4,84,84)}$ — 4 frames gris apilados.

{{< callout type="info" >}}
**Frame stack ≠ frame skip.** Se parecen los nombres pero son ortogonales: **skip** = cada cuánto *decides* (eficiencia temporal); **stack** = cuántos frames *ves a la vez* (para percibir movimiento). Combinados, cada uno de los 4 frames apilados está separado por 4 frames de juego → el agente ve ~16 frames de historia efectiva.
{{< /callout >}}

### El orden de composición es el algoritmo

Los wrappers se anidan como capas de cebolla, y el orden **no es arbitrario**:

```python
def make_env(env_name, version=2, max_steps=5000):
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)   # el max opera sobre frames CRUDOS
    env = WarpFrame(env, version=version)  # recién ahora reduce resolución
    env = FrameStack(env, 4)           # apila frames YA preprocesados
    return env
```

Dos decisiones clave en `gym.make`:
- **`frameskip=1`** ("NoFrameskip"): el skip lo maneja `MaxAndSkipEnv` manualmente. Si ALE también hiciera skip, sería skip de 16 — un desastre.
- **`repeat_action_probability=0.0`**: desactiva las "sticky actions" → entorno **determinista**, más reproducible.

El parámetro `version=2` de `WarpFrame` activa un **recorte** `frame[13:98]` que elimina el marcador (score) y centra la cancha. Esto **debe coincidir** con cómo fue entrenado el experto (`Expert2.model`): si el experto vio imágenes recortadas y le pasas imágenes sin recortar, sus predicciones se degradan sin ningún error visible. La versión del preprocesamiento y el archivo del experto están **acoplados**.

Al instanciar el entorno, el notebook verifica los espacios:

```
Obs space: Box(0.0, 1.0, (4, 84, 84), float32)
Act space: Discrete(4)
['NOOP', 'FIRE', 'RIGHT', 'LEFT']
```

Cuatro acciones discretas → el estudiante será un **clasificador de 4 clases**.

## El modelo: la CNN de DQN

Experto y estudiante comparten la **misma arquitectura** (la de Mnih et al. 2015): 3 capas convolucionales + 2 densas.

```python
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self.get_conv_out_size(input_shape)  # calcula 3136 con forward dummy
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        if x.dim() == 5:            # (B,4,84,84,1) -> (B,4,84,84)
            x = x.squeeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)          # logits crudos, SIN softmax
```

Flujo dimensional: $(4,84,84) \to (32,20,20) \to (64,9,9) \to (64,7,7) \to \text{flatten}(3136) \to 512 \to 4$.

Tres detalles no triviales:

1. **`get_conv_out_size` calcula el 3136 con un forward "en seco"** sobre un tensor de ceros, en vez de hardcodearlo. Robusto: si cambias el tamaño de entrada, el número se recalcula solo.
2. **Sin max-pooling.** DQN usa *stride* para reducir resolución, no pooling. El pooling introduce invarianza a la traslación, pero en un juego *la posición exacta de la pelota importa* — no la quieres invariante.
3. **`if x.dim() == 5: squeeze(-1)`** es un parche real: `WarpFrame` devuelve frames `(84,84,1)` y `FrameStack` los apila con un `1` sobrante. Los outputs del notebook lo confirman: `Observation size: torch.Size([4, 84, 84, 1])`. El parche **sí se activa** en cada forward.

{{< callout type="info" >}}
**Misma red, dos semánticas.** El `forward` termina en `fc2` **sin softmax** — emite 4 números reales sin comprometer una interpretación. El **experto** los lee como **Q-values** (retorno futuro esperado, entrenado con Q-learning); el **estudiante** los entrena como **logits** que `cross_entropy` normaliza a probabilidades. Esta dualidad responde las preguntas 3 y 4 de la tarea.
{{< /callout >}}

## El experto: cargar los pesos

El experto es un DQN pre-entrenado. Cargarlo es el patrón canónico de PyTorch — arquitectura + pesos por separado:

```python
expert_state_dict = torch.load('Expert2.model', map_location=DEVICE, weights_only=False)
expert_model = DQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
expert_model.load_state_dict(expert_state_dict)
```

Puntos finos:
- **`map_location=DEVICE`** reubica los tensores al dispositivo actual → el mismo notebook corre en Colab-GPU y en CPU sin cambios.
- **`weights_only=False`** es una **puerta de seguridad**: `torch.load` usa `pickle`, que puede ejecutar código arbitrario. Solo es aceptable porque confías en la fuente (el archivo del curso). En producción, cargar así un archivo de origen no confiable es un riesgo real.
- El `else` del notebook, si el archivo no existe, crea un experto con **pesos random** y solo imprime un WARNING → **fallo silencioso**: el lab "funciona" pero imita basura. Si el baseline del experto sale ridículamente bajo, este es el sospechoso.

## Consultar al experto

Una función mínima extrae la acción de cualquier modelo:

```python
def get_action_from_policy(model, states):
    model.eval()
    with torch.no_grad():                       # sin grafo de autograd: ahorra memoria
        output = model(states)                  # (B, 4)
        _, best_action = torch.max(output, dim=1)  # argmax sobre las 4 acciones
    return best_action
```

Es **agnóstica a la interpretación**: hace argmax sobre 4 números, sean Q-values (experto) o logits (estudiante). Por eso sirve para ambos. El `torch.no_grad()` es crítico: en el loop de DAGGER esta función se llama decenas de miles de veces; sin él, acumularías grafos inútiles y te quedarías sin memoria.

Con el entorno, la red y el experto listos, ya se puede construir el bucle de DAGGER — la siguiente parte.
