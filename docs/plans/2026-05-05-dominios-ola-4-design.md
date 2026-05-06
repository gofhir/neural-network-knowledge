---
title: "Diseño — Sección Dominios, Ola 4 (Video)"
date: 2026-05-05
status: aprobado
autor: Roberto Araneda
---

# Diseño — Ola 4: Video

## Contexto

Ola 1 (mergeada 2026-05-03) entregó la infraestructura + Texto/NLP + Visión. Ola 2 (mergeada 2026-05-05) sumó Multimodal. Ola 3 (mergeada 2026-05-05) sumó Audio/Voz. Esta Ola 4 agrega **Video** como quinto dominio completo. Restan tras esta ola: Robótica/RL y Datos estructurados (Ola 5).

## Material existente

**Sin material específico** de video en `fundamentos/` ni `papers/`. Material adyacente disponible (CNN, ResNet, ViT, Transformer, atención) — pero la decisión aprobada es **mantener todos los hitos `minimal`** desde el inicio en lugar de marcar `covered` con enlaces a fundamentos que no discuten la aplicación a video. La lección de Ola 3 (4 issues factuales, dos por `covered` overstated) justifica este enfoque más conservador.

## Decisiones aprobadas

1. Una página: `dominios/video/_index.md`, patrón idéntico a Audio/Multimodal/Texto/Visión.
2. Cinco eras: pre-deep / handcrafted → two-stream y 3D-CNN tempranas → 3D-CNN profundas → Video Transformers → generación + foundation.
3. Aproximadamente 19 hitos distribuidos 3+3+3+4+6.
4. Todos los hitos `minimal` (sin material específico). Sin `deep` ni `covered`.
5. Implementación en 4 tasks (infraestructura ya existe).
6. Branch: `feat/dominios-ola-4` (creada desde `main` post-merge de Ola 3).

## Estructura de la página

`site/content/dominios/video/_index.md` reemplaza el stub. Mismo molde:

1. Front matter (`title: "Video"`, `weight: 4`, `sidebar.open: true`).
2. `# Video` (H1).
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

1. **Video es una secuencia de imágenes con costo computacional explosivo.** Un clip de 16 frames a resolución 224×224 son ~16× los píxeles de una imagen estática. Procesarlo cuadro por cuadro pierde la información de movimiento que define qué está pasando; procesarlo conjuntamente requiere arquitecturas que no escalen cuadráticamente con la longitud temporal. Cada generación de modelos navegó esa tensión de forma distinta.

2. **Tres tensiones definen el campo:** (1) cómo **modelar movimiento sin desperdiciar parámetros** — kernels 3D, two-stream con flujo óptico, factorización espacio-tiempo, (2) cómo **escalar a clips largos** — ventanas de atención, jerarquías, modelos eficientes, y (3) la divergencia entre **comprensión** (action recognition, video classification, donde el ground truth es una etiqueta) y **generación** (texto-a-video, donde la frontera es la coherencia temporal extendida y la física aproximada). Hasta 2022 estos eran problemas separados; los modelos generativos actuales (Sora, Veo) están empezando a unificarlos.

## Línea de tiempo — eras y hitos

### Era 1 — Pre-deep / handcrafted (2003-2012)

| Hito | Año | Status |
|---|---|---|
| HOG3D / Cuboids 3D / SIFT 3D | 2008 | `minimal` |
| Dense Trajectories (Wang & Schmid) | 2011 | `minimal` |
| iDT — improved Dense Trajectories | 2013 | `minimal` |

### Era 2 — Two-stream y 3D-CNN tempranas (2014-2015)

| Hito | Año | Status |
|---|---|---|
| Karpathy CVPR (slow/fast fusion) | 2014 | `minimal` |
| Two-Stream (Simonyan & Zisserman) | 2014 | `minimal` |
| C3D (Tran et al.) | 2015 | `minimal` |

### Era 3 — 3D-CNN profundas (2017-2019)

| Hito | Año | Status |
|---|---|---|
| I3D (Carreira & Zisserman / DeepMind) | 2017 | `minimal` |
| R(2+1)D (Tran et al. / FAIR) | 2018 | `minimal` |
| SlowFast (Feichtenhofer et al. / FAIR) | 2019 | `minimal` |

### Era 4 — Video Transformers (2021-2022)

| Hito | Año | Status |
|---|---|---|
| TimeSformer (Bertasius et al. / FAIR) | 2021 | `minimal` |
| ViViT (Arnab et al. / Google) | 2021 | `minimal` |
| MViT (Fan et al. / FAIR) | 2021 | `minimal` |
| Video Swin (Liu et al.) | 2022 | `minimal` |

### Era 5 — Generación + foundation (2022-presente)

| Hito | Año | Status |
|---|---|---|
| Make-A-Video (Singer / Meta) | 2022 | `minimal` |
| Imagen Video (Ho / Google) | 2022 | `minimal` |
| Stable Video Diffusion (Stability) | 2023 | `minimal` |
| Sora (OpenAI) | 2024 | `minimal` |
| Veo (Google DeepMind) | 2024 | `minimal` |
| Kling (Kuaishou) / Runway Gen-3 | 2024 | `minimal` |

**Total: 19 hitos** (3+3+3+4+6).

## Estado del arte hoy — esbozo

Callout con modelos punteros 2024-2025:

- **Sora 2** — OpenAI. Generación de video con coherencia temporal extendida (60s+).
- **Veo 2** — Google DeepMind. Video en alta resolución con prompt complejo.
- **Kling v2** — Kuaishou. Líder en China; calidad competitiva con frontier occidental.
- **Runway Gen-4 / Gen-3 Alpha** — producción para creadores.
- **Stable Video Diffusion 3** — Stability AI. Open weights para video corto.
- **Pika 2.0** — generación de video con control de movimiento.
- **Comprensión**: GPT-4o / Gemini 2.5 / Claude analizan video largo nativamente.

## Casos de uso reales

- **Acción y video clasificación**: Kinetics, Something-Something — pre-2022 dominaban CNN 3D.
- **Generación de video corto**: Sora, Runway, Pika, Kling — marketing, redes sociales, previsualización.
- **VFX y postproducción**: integración en pipelines de Hollywood (Wonder Studio, Runway).
- **Vigilancia y análisis de seguridad**: detección de eventos, action recognition en CCTV.
- **Análisis deportivo**: tracking, generación de highlights automáticos.
- **Conducción autónoma**: análisis de video en tiempo real para predicción de trayectorias.
- **Comprensión de video largo**: resumen automático de reuniones, podcasts, lectures.

## Qué viene

- **Coherencia física genuina** en generación (más allá de Sora — conservación de masa, causalidad, identidad).
- **Video largo generativo** (5+ minutos coherentes con narrativa).
- **Vision-Language-Action** para robótica (RT-2, π0 — pendientes de Ola 5).
- **Edición de video por prompt** (modificar contenido existente, no solo generar nuevo).
- **Modelos eficientes** para edge / móvil.
- **Detección de deepfakes de video** como contramedida industrial.

## Plan de implementación (4 tasks)

| Task | Entregable |
|---|---|
| 1 | Front matter + problema central + timeline (5 eras + 19 hitos) |
| 2 | 5 era subsections |
| 3 | SOTA + casos + qué viene + recursos |
| 4 | Build limpio + push + PR |

Sin tocar shortcodes, CSS, menú ni stats.

## Convenciones (heredadas)

- Español con tildes correctas.
- Tono pedagógico-narrativo.
- 800-1500 palabras totales.
- Sin Co-Authored-By en commits.
- `weight: 4` (orden actual del sidebar).

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Datos puntuales (años, autores) sin verificar | Code reviewer subagent debe verificar como en Olas previas. Las fechas de Era 5 (Sora, Veo, Kling, Pika) cambian rápido — usar arXiv preprint year cuando sea posible |
| Solapamiento con Era 5 de Visión y Multimodal (Sora, Stable Video Diffusion) | Intencional. Cada página los enmarca desde su ángulo |
| Era 5 con 6 hitos podría exceder lo razonable | Aceptable — la era actual del dominio justifica más densidad |
