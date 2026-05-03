---
title: "Diseño — Sección Dominios"
date: 2026-05-03
status: aprobado
autor: Roberto Araneda
---

# Diseño — Sección "Dominios"

## Contexto y motivación

El site del Diplomado IA UC tiene actualmente cuatro secciones de primer nivel: **Fundamentos** (teoría organizada por tema), **Clases** (apuntes cronológicos), **Laboratorios** (notebooks) y **Papers** (análisis). Estas vistas son temáticas o cronológicas, pero no cuentan la historia evolutiva por **dominio de aplicación**.

Una persona interesada en NLP debería poder seguir un recorrido natural — desde n-gramas y word2vec hasta los LLMs actuales — entendiendo *por qué* surgió cada arquitectura y *qué limitación heredó* de la anterior. Lo mismo para visión, audio, video, multimodal, robótica/RL y datos estructurados.

La sección **Dominios** llena ese vacío. Es una vista transversal y narrativa que reaprovecha el material existente en Fundamentos y Papers, agregando solo el conector histórico y los hitos faltantes.

## Decisiones aprobadas

1. **Sección nueva**, no reemplaza nada. Se agrega como quinto pilar del menú principal.
2. **Siete dominios**: Texto/NLP, Visión, Audio/Voz, Video, Multimodal, Robótica/RL, Datos estructurados.
3. **Nombre de la sección:** "Dominios".
4. **Estructura interna por dominio:** híbrida — línea de tiempo visual con eras + cards de hitos clickeables que enlazan a Fundamentos/Papers existentes.
5. **Línea de tiempo visual:** shortcode Hugo custom (HTML/CSS puro, sin JS).
6. **Política de hitos sin material:** tres estados — `deep` (Fundamento dedicado), `covered` (mencionado en otro Fundamento/Paper), `minimal` (descripción inline 2-3 líneas, sin enlace externo).
7. **Implementación en olas:** Ola 1 = infraestructura + Texto + Visión. Olas siguientes agregan Multimodal, Audio, Video, Robótica/RL, Estructurados.
8. **Idioma:** español con tildes correctas en todo el contenido nuevo.

## Arquitectura de información

### Estructura de archivos

```
site/content/dominios/
├── _index.md                    # landing: grid de 7 dominios
├── texto/_index.md              # Texto / NLP
├── vision/_index.md             # Visión
├── audio/_index.md              # Audio / Voz
├── video/_index.md              # Video
├── multimodal/_index.md         # Multimodal
├── robotica/_index.md           # Robótica / RL
└── estructurados/_index.md      # Datos estructurados
```

### Menú principal

En `site/hugo.yaml` se inserta una nueva entrada **"Dominios"** entre Fundamentos y Clases con `weight: 15`. El orden de lectura natural queda:

1. Fundamentos (10)
2. Dominios (15)
3. Clases (20)
4. Laboratorios (30)
5. Papers (40)

### Landing `dominios/_index.md`

- Introducción breve (1-2 párrafos): "Una vista transversal de cómo cada modalidad de datos forzó la evolución de las arquitecturas".
- Grid de 7 cards con `feature-grid` de Hextra (mismo componente que la home), una card por dominio con icono Hextra + 1 línea descriptiva + enlace.

## Plantilla de página de dominio

Cada `dominios/<dominio>/_index.md` sigue **exactamente** este orden de secciones:

1. **Front matter:** `title`, `weight` (1=Texto, 2=Visión, 3=Audio, 4=Video, 5=Multimodal, 6=Robótica, 7=Estructurados), `sidebar.open: true`, `type: docs`.
2. **El problema central del dominio** — 1-2 párrafos sobre qué hace especial el dato (continuidad temporal en audio, jerarquía espacial en imagen, composicionalidad discreta en texto, etc.) y cómo esas propiedades fuerzan decisiones arquitectónicas.
3. **Línea de tiempo visual** — invocación del shortcode `{{< timeline >}}`.
4. **Las eras explicadas** — una subsección `## Era N — <nombre>` por era, con la estructura fija:
   - **Problema heredado** (1 párrafo).
   - **Idea clave** (1 párrafo).
   - **Hitos** (cards de modelos vía shortcode).
   - **Qué la destronó** (1 párrafo) — la última era omite esta subsección.
5. **Estado del arte hoy** — caja `{{< callout type="info" >}}` con 3-5 modelos punteros 2024-2025, cada uno con 2-3 líneas y enlace al paper.
6. **Casos de uso reales** — lista de 3-5 ejemplos en producción/industria.
7. **Qué viene** — 1 párrafo especulativo. Se cierra con "Última actualización: YYYY-MM-DD".
8. **Recursos relacionados** — enlaces internos a Fundamentos, Papers y Clases del dominio.

### Convenciones de escritura

- Tono pedagógico-narrativo, no enciclopédico. Cada era cuenta una historia: problema → idea → modelos → caída.
- Nunca duplicar contenido de Fundamentos o Papers; siempre enlazar.
- Largo objetivo por página: **800-1500 palabras** sin contar la timeline. Si crece más, se parte el exceso en un Fundamento nuevo.
- Tildes correctas en todo el contenido nuevo.

## Componentes Hugo nuevos

### `site/layouts/shortcodes/timeline.html`

Wrapper de la línea de tiempo. Sintaxis de uso:

```go-html-template
{{< timeline >}}
  {{< era name="Era pre-neural" years="1948-2010" >}}
    {{< hito year="1948" name="n-gramas" status="minimal" >}}
      Modelo de lenguaje basado en frecuencias de secuencias cortas.
      **Por qué importó:** estableció el problema de predecir la siguiente palabra.
    {{< /hito >}}
    {{< hito year="2003" name="Bengio NNLM" status="covered" link="/fundamentos/embeddings-distribuidos" >}}
      Primera red neuronal para modelar lenguaje; introduce embeddings densos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de embeddings" years="2013-2017" >}}
    ...
  {{< /era >}}
{{< /timeline >}}
```

### `site/layouts/shortcodes/era.html`

Encabezado de era — rango de años + nombre — y contenedor de los hitos hijos.

### `site/layouts/shortcodes/hito.html`

Card individual. El atributo `status` controla comportamiento y estilo:

| `status` | Comportamiento | Estilo |
|----------|----------------|--------|
| `deep` | Card clickeable; `link` apunta a un Fundamento dedicado. Año + nombre + 1 línea + ícono "leer más". | Border sólido, hover destacado |
| `covered` | Card clickeable; `link` apunta a Fundamento o Paper donde se menciona (puede usar ancla). Año + nombre + 1 línea. | Border más suave |
| `minimal` | Card no clickeable. Año + nombre + 2-3 líneas inline. | Border punteado, fondo levemente distinto |

### `site/assets/css/timeline.css` (o equivalente)

- Layout vertical: eras apiladas verticalmente. Encabezado de era (rango de años + nombre) y debajo grilla de cards.
- Grilla: 3 columnas en desktop, 1 en móvil.
- Línea vertical decorativa a la izquierda con un punto por era ("track" temporal).
- Tema light/dark: usar variables CSS de Hextra (`--hextra-content-bg`, `--hextra-primary`, etc.) para respetar el toggle automáticamente.
- Plan B si Hextra no expone hook limpio para CSS custom: `<style>` inline dentro del shortcode con las mismas variables.

## Plan de contenido por olas

### Ola 1 — Infraestructura + 2 dominios completos

| Entregable | Detalle |
|---|---|
| Shortcodes | `timeline.html`, `era.html`, `hito.html` + CSS |
| Menú | Entrada "Dominios" en `hugo.yaml` (weight 15) |
| Landing | `dominios/_index.md` con grid de 7 cards (las 5 olas posteriores enlazan a páginas stub) |
| **Texto / NLP** | Página completa: 5 eras (pre-neural → embeddings → recurrente → atención → LLMs), casos de uso, SOTA |
| **Visión** | Página completa: 5 eras (pre-neural → CNN → más profundas/residuales → ViT → multimodal), casos de uso, SOTA |
| Stats home | Actualizar `content/_index.md` con bloque "7 Dominios" |

### Ola 2 — Multimodal

Dominio con material parcial existente (CLIP, show-and-tell, show-attend-tell, foundation-models). Eras planeadas: pre-multimodal → captioning con atención → CLIP/contrastivo → VLMs → generación texto-imagen.

### Ola 3 — Audio / Voz y Video

Sin material existente, escritos desde cero.

- **Audio:** acústica clásica → DNN-HMM → end-to-end (CTC, RNN-T) → transformers de audio → fundacionales. Hitos: MFCC, DeepSpeech, wav2vec, Whisper, Tacotron, VALL-E.
- **Video:** cuadro-a-cuadro → two-stream → 3D-CNN → video transformers → generación. Hitos: I3D, SlowFast, ViViT, Sora.

### Ola 4 — Robótica/RL y Estructurados

- **Robótica/RL:** RL clásico → Deep RL → AlphaGo era → RLHF → robot foundation models. Hitos: Q-learning, DQN, AlphaGo/Zero, PPO, RT-2, π0.
- **Estructurados:** GBM clásico → DL para tabular → series temporales DL → grafos → fundacionales tabulares. Hitos: XGBoost, TabNet, FT-Transformer, DeepAR, N-BEATS, GCN, GAT, TabPFN.

### Cards stub de las olas pendientes

Desde el día 1, las 7 cards del landing existen y son clickeables. Las olas 2-4 abren a páginas mínimas (`_index.md` con un párrafo introductorio y placeholder "Página en construcción — disponible en próxima ola"). Esto evita 404s y mantiene la sección completa estructuralmente.

## Pruebas y verificación

- `hugo --gc --minify` debe compilar sin warnings.
- Inspección visual de las 2 páginas de Ola 1 en desktop y móvil, en light y dark mode.
- Búsqueda FlexSearch debe indexar las páginas nuevas (verificar buscando "word2vec" o "Transformer" desde la home).
- Enlaces internos no rotos: correr `htmltest` o `lychee` sobre `public/` (a integrar en el plan de implementación).

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Shortcode timeline rompe en builds GitHub Pages | Implementar y previsualizar localmente antes de mergear |
| Páginas crecen y duplican Fundamentos | Regla dura: hitos `minimal` máx. 3 líneas; explicaciones más largas se promueven a Fundamento dedicado y el hito pasa a `deep` |
| "Estado del arte" envejece rápido | Fecha de última actualización visible al pie + recordatorio cada 6 meses |
| Hextra no expone hooks limpios de CSS | Plan B: `<style>` inline en el shortcode con variables CSS de Hextra |

## Próximos pasos

1. Commit de este documento de diseño.
2. Generar plan de implementación detallado de la **Ola 1** vía la skill `superpowers:writing-plans` en `docs/plans/2026-05-03-dominios-ola-1-plan.md`.
3. Ejecutar la implementación en una nueva sesión guiada por ese plan.
