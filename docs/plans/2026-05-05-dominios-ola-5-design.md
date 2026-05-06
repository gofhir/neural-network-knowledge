---
title: "Diseño — Sección Dominios, Ola 5 (Robótica / RL)"
date: 2026-05-05
status: aprobado
autor: Roberto Araneda
---

# Diseño — Ola 5: Robótica / RL

## Contexto

Olas 1-4 entregaron Texto/NLP, Visión, Multimodal, Audio/Voz, Video. Esta Ola 5 agrega **Robótica / RL** como sexto dominio. Resta tras esta ola: Datos estructurados (Ola 6 final).

## Material existente

**Material adyacente disponible** (especialmente fuerte para RLHF/DPO):
- `/fundamentos/sft` — cubre RLHF en profundidad (ya validado en Texto/NLP).
- `/fundamentos/dpo` — cubre DPO (Rafailov et al., 2023).
- `/fundamentos/kl-implicito` — el mecanismo formal detrás de DPO.
- `/fundamentos/bradley-terry` — modelo de preferencias usado en RLHF.
- `/fundamentos/foundation-models` — contexto para los robot foundation models.

Esto justifica 2 hitos `covered` en la Era 4 (InstructGPT/RLHF → `sft`, DPO → `dpo`). Resto de hitos `minimal`.

## Decisiones aprobadas

1. Una página: `dominios/robotica/_index.md`, patrón idéntico a Audio/Video.
2. Cinco eras: RL clásico → Deep RL temprano → era AlphaGo → PPO+RLHF → Robot foundation models.
3. Aproximadamente 19 hitos distribuidos 3+3+4+4+5.
4. Status mix: 0 deep + 2 covered + 17 minimal.
5. Implementación en 4 tasks (infraestructura ya existe).
6. Branch: `feat/dominios-ola-5`.

## Estructura de la página

`site/content/dominios/robotica/_index.md` reemplaza el stub. Mismo molde:

1. Front matter (`title: "Robótica / RL"`, `weight: 6`, `sidebar.open: true`).
2. `# Robótica / RL` (H1).
3. `## El problema central` — 1-2 párrafos.
4. `## Línea de tiempo` con 5 eras y ~19 hitos.
5. 5 subsecciones `## Era N — <nombre> (<rango>)` con Problema heredado / Idea clave / Qué la destronó (eras 1-4) o Qué viene (era 5).
6. `## Estado del arte hoy` (callout).
7. `## Casos de uso reales`.
8. `## Qué viene`.
9. `## Recursos relacionados`.
10. Pie con fecha.

## El problema central — esbozo

Dos párrafos:

1. **Decisiones secuenciales bajo recompensa.** A diferencia de los demás dominios (texto, imagen, audio, video) donde el modelo recibe entrada y emite salida en un paso, RL implica un agente que actúa repetidamente en un entorno: cada acción cambia el estado, el entorno responde con una recompensa escalar, y el objetivo es maximizar la recompensa acumulada a lo largo del tiempo. Esto introduce dependencias temporales largas, exploración del espacio de acciones, y la pregunta del *credit assignment* — qué decisión pasada es responsable de la recompensa actual.

2. **Tres tensiones específicas:** (1) **exploración vs explotación** — el agente debe descubrir acciones nuevas sin perder las que ya sabe que funcionan; (2) **eficiencia de muestra** — RL clásico necesita millones de interacciones, lo cual es viable en simulación pero infactible en robótica física donde cada interacción cuesta tiempo y desgaste; (3) **alineamiento** — cómo formular la "recompensa" cuando viene de preferencias humanas (RLHF), de instrucciones en lenguaje natural (robot foundation models), o cuando el espacio de objetivos es demasiado complejo para una función numérica fija.

## Línea de tiempo — eras y hitos

### Era 1 — RL clásico (1989-2010)

| Hito | Año | Status |
|---|---|---|
| Q-learning (Watkins) | 1989 | `minimal` |
| SARSA / TD-learning | 1994 | `minimal` |
| Policy gradients / REINFORCE (Williams) | 1992 | `minimal` |

### Era 2 — Deep RL temprano (2013-2016)

| Hito | Año | Status |
|---|---|---|
| DQN (Mnih et al. / DeepMind) | 2013/2015 | `minimal` |
| A3C (Mnih et al.) | 2016 | `minimal` |
| DDPG (Lillicrap et al.) | 2015 | `minimal` |

### Era 3 — La era AlphaGo (2016-2019)

| Hito | Año | Status |
|---|---|---|
| AlphaGo (Silver et al. / DeepMind) | 2016 | `minimal` |
| AlphaZero (DeepMind) | 2017 | `minimal` |
| AlphaStar (StarCraft II) | 2019 | `minimal` |
| MuZero (DeepMind) | 2019 | `minimal` |

### Era 4 — PPO + RLHF (2017-2023)

| Hito | Año | Status |
|---|---|---|
| PPO (Schulman et al. / OpenAI) | 2017 | `minimal` |
| InstructGPT / RLHF | 2022 | `covered` → `/fundamentos/sft` |
| RLAIF / Constitutional AI (Anthropic) | 2022-2023 | `minimal` |
| DPO (Rafailov et al.) | 2023 | `covered` → `/fundamentos/dpo` |

### Era 5 — Robot foundation models (2022-presente)

| Hito | Año | Status |
|---|---|---|
| SayCan (Google) | 2022 | `minimal` |
| RT-1 (Google) | 2022 | `minimal` |
| RT-2 (Google DeepMind) | 2023 | `minimal` |
| OpenVLA (Stanford) | 2024 | `minimal` |
| π0 (Physical Intelligence) | 2024 | `minimal` |

**Total: 18 hitos** (3+3+4+4+5). Decidí omitir Gemini Robotics 2025 (incluida en SOTA pero no como hito de timeline porque la era ya cierra con π0/OpenVLA cubriendo el patrón VLA).

## Estado del arte hoy — esbozo

Callout con frontier 2024-2025:

- **Gemini Robotics** — Google DeepMind. VLA (Vision-Language-Action) integrado a Gemini 2.5.
- **π0 / Physical Intelligence** — robot foundation model multi-embodiment con destreza fina.
- **OpenVLA** — open-source, 7B parámetros, base de muchas pipelines downstream.
- **NVIDIA GR00T** — humanoid foundation model.
- **Claude Computer Use / GPT-4o agents** — agentes que actúan sobre interfaces gráficas (no robótica física, pero RL del mundo digital).
- **DeepSeek-R1** — modelo de razonamiento entrenado con RL puro sobre cadenas de pensamiento.
- **Tesla Optimus / Figure 02** — humanoides en producción con foundation models.

## Casos de uso reales

- **Robótica industrial**: pick-and-place, control de calidad, ensamblado en líneas con RT-X.
- **Robots humanoides**: Tesla Optimus, Figure, Apptronik — manipulación general en almacenes y fábricas.
- **Asistentes domésticos**: 1X, Boston Dynamics — limpieza, asistencia personal.
- **Conducción autónoma**: Tesla FSD, Waymo, Cruise — RL/IL para predicción y planning.
- **Juegos**: AlphaGo, AlphaStar, OpenAI Five — superhumanos en juegos cerrados.
- **Asistentes conversacionales alineados**: ChatGPT, Claude, Gemini — RLHF/DPO en cada release.
- **Agentes digitales**: Claude Computer Use, ChatGPT Operator — automatizar tareas en interfaces.
- **Optimización de procesos**: data centers (Google), logística, trading.
- **Cirugía robótica**: Da Vinci con asistencia inteligente, sub-tareas autónomas.

## Qué viene

- **Multi-embodiment** (un modelo controla brazos, humanoides, manipuladores móviles, drones).
- **Generalización por lenguaje** (instrucciones complejas en lenguaje natural).
- **Sim-to-real** masivo (entrenar en simulación masiva, transferir a físico).
- **Razonamiento jerárquico** (planeación de largo horizonte + control de bajo nivel).
- **Recompensas aprendidas** (no diseñadas a mano — modelos de recompensa de visión-lenguaje).
- **RL de razonamiento** (post-DeepSeek-R1, los LLMs aprenden a razonar via RL puro).
- **Robotic data flywheel** (despliegue → recolección → entrenamiento → mejor modelo).

## Plan de implementación (4 tasks)

| Task | Entregable |
|---|---|
| 1 | Front matter + problema central + timeline (5 eras + 18 hitos) |
| 2 | 5 era subsections |
| 3 | SOTA + casos + qué viene + recursos |
| 4 | Build limpio + push + PR |

Sin tocar shortcodes, CSS, menú ni stats.

## Convenciones (heredadas)

- Español con tildes correctas.
- Tono pedagógico-narrativo.
- 800-1500 palabras totales.
- Sin Co-Authored-By en commits.
- `weight: 6` (orden actual del sidebar).

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Datos puntuales (años, autores) sin verificar | Code reviewer subagent debe validar especialmente fechas/orden de DQN (2013 NIPS workshop vs 2015 Nature paper), RT-1/RT-2/OpenVLA |
| `covered` para InstructGPT y DPO podría ser overstated | Validado en Olas previas: `sft.md` y `dpo.md` cubren ambos en profundidad. OK. |
| Era 4 mezcla RL puro (PPO) con alineamiento (RLHF/DPO) — distintos paradigmas | Aceptable: ambos comparten la maquinaria de policy optimization; la prosa de Era 4 lo aclara |
| Solapamiento con Era 5 de Texto/NLP (SFT, DPO, RLHF) | Intencional. Cada página los enmarca desde su ángulo (Texto: como alineamiento de LLMs; Robótica: como variante de policy optimization) |
