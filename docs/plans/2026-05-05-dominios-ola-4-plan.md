# Sección Dominios — Ola 4 (Video) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reemplazar el stub `dominios/video/` por una página completa que narre la evolución desde features handcrafted (HOG3D, Dense Trajectories, 2008-2013) hasta los modelos generativos de video frontera 2024-2025 (Sora, Veo, Kling, Runway Gen-3).

**Architecture:** Una página Markdown construida en 3 commits siguiendo el patrón de Audio (Ola 3): Task 1 = front matter + intro + timeline; Task 2 = era subsections; Task 3 = SOTA + casos + recursos. Toda la infraestructura (shortcodes, CSS, menú) ya existe en `main` post-Ola 3.

**Tech Stack:** Hugo + tema Hextra (vendored vía `go.mod`), Markdown con shortcodes Hugo, KaTeX inline, FlexSearch. baseURL: `/neural-network-knowledge/`.

**Diseño de referencia:** [docs/plans/2026-05-05-dominios-ola-4-design.md](2026-05-05-dominios-ola-4-design.md).

**Convenciones del codebase verificadas:**
- Shortcodes ya disponibles: `{{< timeline >}}`, `{{< era >}}`, `{{< hito >}}`. CSS soporta light/dark + responsive.
- Status taxonomy: `deep` (Fundamento dedicado), `covered` (mencionado en otro Fundamento/Paper), `minimal` (descripción inline).
- Front matter: `title`, `weight: 4`, `sidebar.open: true`. `type: docs` cascadea.
- `{{< callout type="info" >}}` para SOTA box.
- Sin Co-Authored-By en commits.
- Español con tildes correctas.

**Working directory:** `/Users/robertoaraneda/projects/personal/courses/ia-uc/`. **Branch:** `feat/dominios-ola-4`.

**Comando de build:** `cd site && hugo --gc --minify`.

**Decisión de status:** **Todos los hitos son `minimal`**. Sin material específico de video en `fundamentos/` ni `papers/`; los enlaces a CNN/ViT/Transformer adyacentes serían "overstated covered" como ocurrió en la review de Audio (4 fixes post-review). Más honesto y rápido ir directo a `minimal`.

**Stub actual de `site/content/dominios/video/_index.md`** (heredado de Ola 1):
```markdown
---
title: "Video"
weight: 4
sidebar:
  open: true
---

# Video

El reto de combinar espacio y tiempo: two-stream, 3D-CNN, video transformers y generación temporal coherente.

> **Página en construcción.** Esta sección estará disponible en una próxima ola de la sección Dominios. Ver el plan en [docs/plans/2026-05-03-dominios-design.md](https://github.com/robertoaraneda/diplomado-ia-uc/blob/main/docs/plans/2026-05-03-dominios-design.md).
```

Task 1 lo sobrescribe completo.

---

## Task 1: Video — front matter + problema central + línea de tiempo

**Files:**
- Modify: `site/content/dominios/video/_index.md` (overwrite stub completo)

**Step 1: Verificar branch**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git branch --show-current  # must be feat/dominios-ola-4
```

If not on `feat/dominios-ola-4`, stop and report.

**Step 2: Sobrescribir el stub con EXACTAMENTE este contenido**

Verify all 19 `{{< hito ... >}}` opening tags have a matching `{{< /hito >}}` closing tag — Hugo will fail if any tag is malformed.

```markdown
---
title: "Video"
weight: 4
sidebar:
  open: true
---

# Video

## El problema central

Video es una **secuencia de imágenes con costo computacional explosivo**. Un clip de 16 frames a resolución 224×224 son ~16× los píxeles de una imagen estática, y ese factor crece linealmente con la duración. Procesar el video cuadro por cuadro pierde la información de **movimiento** que define qué está pasando — *correr* y *trotar* tienen los mismos pixels promedio pero dinámicas distintas. Procesarlo conjuntamente requiere arquitecturas que no escalen cuadráticamente con la longitud temporal. Cada generación de modelos navegó esa tensión de forma distinta.

Tres tensiones definen el campo: (1) cómo **modelar movimiento sin desperdiciar parámetros** — kernels convolucionales 3D, two-stream con flujo óptico, factorización espacio-tiempo; (2) cómo **escalar a clips largos** — ventanas de atención, jerarquías, modelos eficientes; y (3) la divergencia entre **comprensión** (action recognition, video classification, donde el ground truth es una etiqueta) y **generación** (texto-a-video, donde la frontera es la coherencia temporal extendida y la física aproximada). Hasta 2022 estos eran problemas separados con arquitecturas dedicadas; los modelos generativos actuales (Sora, Veo) están empezando a unificarlos.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-deep / handcrafted" years="2003-2013" >}}
    {{< hito year="2008" name="HOG3D / Cuboids 3D" status="minimal" >}}
      Klaeser et al., Laptev: extensiones 3D de descriptores de imagen (HOG, SIFT) sobre volúmenes espacio-temporales. **Por qué importó:** primer intento sistemático de capturar movimiento como una feature computable, base de los pipelines clásicos de action recognition.
    {{< /hito >}}
    {{< hito year="2011" name="Dense Trajectories" status="minimal" >}}
      Wang & Schmid (INRIA): trayectorias densas obtenidas siguiendo puntos vía flujo óptico, descritas con HOG/HOF/MBH. **Por qué importó:** estado del arte en HMDB y UCF-101 durante varios años con descriptores hechos a mano.
    {{< /hito >}}
    {{< hito year="2013" name="iDT — improved Dense Trajectories" status="minimal" >}}
      Wang & Schmid: refinamiento con compensación de movimiento de cámara y Fisher Vectors. **Por qué importó:** seguía superando o igualando a las primeras CNNs de video durante 2014-2015 — el handcrafted no se rindió fácil.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era two-stream y 3D-CNN tempranas" years="2014-2015" >}}
    {{< hito year="2014" name="Karpathy CVPR (slow/fast fusion)" status="minimal" >}}
      Karpathy et al. (Google/Stanford): CNNs aplicadas a video con varias estrategias de fusión temporal (early, late, slow, single-frame). **Por qué importó:** primer estudio sistemático de cómo extender CNN-2D a video; resultado descorazonador (igual a single-frame), motivó two-stream.
    {{< /hito >}}
    {{< hito year="2014" name="Two-Stream" status="minimal" >}}
      Simonyan & Zisserman (Oxford): dos CNNs paralelas — una sobre RGB (apariencia), otra sobre flujo óptico (movimiento) — fusionadas en la decisión final. **Por qué importó:** superó a iDT y fijó el patrón "modelar movimiento explícitamente" durante 4 años.
    {{< /hito >}}
    {{< hito year="2015" name="C3D" status="minimal" >}}
      Tran et al. (FAIR): convoluciones 3D (3×3×3) aprendidas extremo a extremo sobre clips de 16 frames. **Por qué importó:** primer modelo 3D-CNN entrenable sobre video sin recurrir a flujo óptico precomputado.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de 3D-CNN profundas" years="2017-2019" >}}
    {{< hito year="2017" name="I3D" status="minimal" >}}
      Carreira & Zisserman (DeepMind): *Inflated 3D ConvNet* — toma una CNN-2D preentrenada en ImageNet (Inception) y "infla" sus filtros 2×2 a 2×2×2, transfiriendo la representación visual a video. Entrenado sobre el nuevo Kinetics-400. **Por qué importó:** el primer modelo que destronó claramente a iDT y two-stream en todos los benchmarks; backbone estándar 2017-2019.
    {{< /hito >}}
    {{< hito year="2018" name="R(2+1)D" status="minimal" >}}
      Tran et al. (FAIR): factoriza convolución 3D en una espacial 2D seguida de una temporal 1D. **Por qué importó:** mejor accuracy con menos parámetros que 3D pura; demostró que separar espacio y tiempo era pedagógicamente útil para el modelo.
    {{< /hito >}}
    {{< hito year="2019" name="SlowFast" status="minimal" >}}
      Feichtenhofer et al. (FAIR): dos vías paralelas — una "slow" a baja frecuencia para apariencia, una "fast" a alta frecuencia para movimiento. Inspirado en M/P pathways del sistema visual. **Por qué importó:** estado del arte en Kinetics y AVA; influyó la división persistente "spatial vs temporal" en arquitecturas posteriores.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Video Transformers" years="2021-2022" >}}
    {{< hito year="2021" name="TimeSformer" status="minimal" >}}
      Bertasius et al. (FAIR): primer Video Transformer puro. Tokeniza cada frame como en ViT y aplica atención factorizada (espacial dentro de cada frame, temporal a través de frames). **Por qué importó:** mostró que ViT podía superar a 3D-CNN profundas con menos cómputo, abriendo la era Transformer en video.
    {{< /hito >}}
    {{< hito year="2021" name="ViViT" status="minimal" >}}
      Arnab et al. (Google): cuatro variantes de Video Transformer con distintos esquemas de factorización espacio-temporal. **Por qué importó:** sistematizó el espacio de diseño de Video Transformers; las opciones "factorized self-attention" y "tubelet embedding" se volvieron estándares.
    {{< /hito >}}
    {{< hito year="2021" name="MViT" status="minimal" >}}
      Fan et al. (FAIR): *Multiscale Vision Transformer*, jerárquico con resolución decreciente y canales crecientes (estilo CNN). **Por qué importó:** Video Transformer más eficiente, base de muchas pipelines downstream.
    {{< /hito >}}
    {{< hito year="2022" name="Video Swin" status="minimal" >}}
      Liu et al.: Swin Transformer extendido a 3D con ventanas espacio-temporales desplazadas. **Por qué importó:** estado del arte en Kinetics-600/700 con sesgo inductivo recuperado vía ventanas locales.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de generación + foundation" years="2022-presente" >}}
    {{< hito year="2022" name="Make-A-Video" status="minimal" >}}
      Singer et al. (Meta): texto-a-video sin pares texto-video etiquetados — usa modelos texto-a-imagen + módulos temporales aprendidos sobre video sin etiquetas. **Por qué importó:** primer texto-a-video de calidad razonable, abrió la frontera generativa.
    {{< /hito >}}
    {{< hito year="2022" name="Imagen Video" status="minimal" >}}
      Ho et al. (Google): cascada de modelos de difusión (base + super-resolución espacial + super-resolución temporal) para generar video 1280×768 a 24 fps. **Por qué importó:** fijó el patrón "diffusion + cascada" para video generativo.
    {{< /hito >}}
    {{< hito year="2023" name="Stable Video Diffusion" status="minimal" >}}
      Stability AI: modelo de difusión latente para video corto, open-source. **Por qué importó:** democratizó la generación de video — open weights que la comunidad pudo extender.
    {{< /hito >}}
    {{< hito year="2024" name="Sora" status="minimal" >}}
      OpenAI (anunciado feb 2024, lanzamiento dic 2024): generación de video de hasta 60 segundos con coherencia temporal extendida, física aproximada y resolución alta. Diffusion sobre tokens de video latentes. **Por qué importó:** salto cualitativo en duración y coherencia; redefinió las expectativas de la frontera.
    {{< /hito >}}
    {{< hito year="2024" name="Veo / Veo 2" status="minimal" >}}
      Google DeepMind (Veo 1 mayo 2024, Veo 2 dic 2024): generación de video en 4K con prompts complejos, control de cámara y estilos cinematográficos. **Por qué importó:** alternativa frontera a Sora con foco en control fino y resolución.
    {{< /hito >}}
    {{< hito year="2024" name="Kling / Runway Gen-3" status="minimal" >}}
      Kuaishou (Kling, jun 2024) y Runway (Gen-3 Alpha, jun 2024): generación de video competitiva con frontier occidental. **Por qué importó:** Kling demostró que China alcanzó paridad rápida; Runway llevó la generación a producción para creadores y estudios.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
```

**Step 3: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

Expected: build clean (only the pre-existing `tabs` deprecation warning is acceptable).

**Step 4: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task1-ola4.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/video/

curl -s -o /tmp/video.html -w "HTTP %{http_code}\n" "$URL"

# H1
grep "<h1[^>]*>Video" /tmp/video.html | head -1

# Sections
grep -c "El problema central" /tmp/video.html
grep -c "Línea de tiempo" /tmp/video.html

# Timeline
grep -c 'class="timeline-container"' /tmp/video.html  # 1

# 5 eras
grep -c 'class="timeline-era"' /tmp/video.html  # 5

# 19 hitos (3+3+3+4+6)
grep -c 'class="timeline-hito timeline-hito-' /tmp/video.html  # 19

# Era headers
grep "Era pre-deep / handcrafted" /tmp/video.html | head -1
grep "Era two-stream y 3D-CNN tempranas" /tmp/video.html | head -1
grep "Era de 3D-CNN profundas" /tmp/video.html | head -1
grep "Era de Video Transformers" /tmp/video.html | head -1
grep "Era de generación + foundation" /tmp/video.html | head -1

# Some hito names
grep "Dense Trajectories" /tmp/video.html | head -1
grep "Two-Stream" /tmp/video.html | head -1
grep "I3D" /tmp/video.html | head -1
grep "TimeSformer" /tmp/video.html | head -1
grep "Sora" /tmp/video.html | head -1

# Status mix (no deep, no covered, all minimal)
grep -c 'class="timeline-hito timeline-hito-deep"' /tmp/video.html  # 0
grep -c 'class="timeline-hito timeline-hito-covered"' /tmp/video.html  # 0
grep -c 'class="timeline-hito timeline-hito-minimal"' /tmp/video.html  # 19

pkill -f "hugo server" || true
sleep 1
```

Expected:
- HTTP 200.
- 1 timeline-container.
- 5 timeline-era.
- 19 hitos (0 deep + 0 covered + 19 minimal).
- All 5 era and key hito names present.

**Step 5: Commit**

```bash
git add site/content/dominios/video/_index.md
git commit -m "feat(dominios/video): problema central + linea de tiempo (5 eras)"
```

NO Co-Authored-By trailer.

---

## Task 2: Video — eras explicadas (5 subsecciones)

**Files:**
- Modify: `site/content/dominios/video/_index.md` (apend al final, después del `{{< /timeline >}}`)

**Step 1: Apender este contenido al final del archivo**

```markdown

## Era 1 — Pre-deep / handcrafted (2003-2013)

### Problema heredado

A inicios de los 2000s la visión por computador clásica había desarrollado descriptores robustos para imágenes (SIFT, HOG, SURF). Pero video era 16+ veces más datos y necesitaba capturar **movimiento** — la información que define qué está pasando. Aplicar SIFT cuadro por cuadro pierde justamente lo importante.

### Idea clave

**Extender los descriptores de imagen a volúmenes espacio-temporales.** HOG3D, Cuboids 3D y SIFT 3D (2008) trataban regiones del video como volúmenes 3D y calculaban descriptores invariantes. Más exitoso aún fue **Dense Trajectories** (Wang & Schmid, 2011): seguir puntos densamente sobre la imagen vía flujo óptico durante varios frames, generando trayectorias que luego se describen con HOG (apariencia), HOF (flujo óptico) y MBH (gradiente del flujo). iDT (2013) refinó la idea compensando el movimiento de cámara y agregando Fisher Vectors para la representación final.

Esa pipeline — handcrafted features + Bag-of-Words + SVM — fue estado del arte en HMDB-51 y UCF-101 durante varios años, e increíblemente seguía superando a las primeras CNNs de video durante 2014-2015.

### Qué la destronó

Los handcrafted features tenían un techo: cada nueva tarea requería diseñar features específicas, y el espacio de patrones espacio-temporales era enorme. Cuando AlexNet había mostrado que CNN-2D aprendían mejores features que SIFT en imágenes, era cuestión de tiempo que pasara lo mismo con video — pero requería más cómputo, más datos etiquetados (Kinetics, 2017) y mejores arquitecturas que la transferencia ingenua de CNN-2D.

## Era 2 — Two-stream y 3D-CNN tempranas (2014-2015)

### Problema heredado

Karpathy et al. (CVPR 2014) hicieron el experimento natural: ¿qué pasa si tomas una CNN-2D entrenada en imágenes y la aplicas a video? Probaron varias estrategias de fusión temporal (early, late, slow, single-frame) sobre Sports-1M. Resultado descorazonador: el modelo single-frame era casi tan bueno como los temporales — la CNN no estaba aprovechando la dimensión temporal del video.

### Idea clave

Dos respuestas paralelas:

1. **Two-Stream** (Simonyan & Zisserman, 2014): si la CNN-2D no aprende movimiento por sí sola, dáselo explícitamente. Una stream procesa RGB (apariencia), otra stream procesa **flujo óptico precomputado** (movimiento). Las predicciones se fusionan al final. Superó a iDT y se volvió el patrón estándar 2014-2018.

2. **C3D** (Tran et al., 2015): convoluciones 3D aprendidas extremo a extremo. Un kernel 3×3×3 procesa simultáneamente espacio y tiempo. Costoso en parámetros y datos, pero conceptualmente más limpio que two-stream — el modelo aprende el movimiento, no se le da gratis.

### Qué la destronó

Two-stream dependía de flujo óptico precomputado (caro, ruidoso, requiere preprocesamiento). C3D era costoso y no escalaba bien en profundidad. Ambos cedieron a la siguiente generación: 3D-CNN inflated desde modelos 2D bien entrenados.

## Era 3 — 3D-CNN profundas (2017-2019)

### Problema heredado

C3D era profundo solo hasta ~8 capas; redes mucho más profundas no entrenaban bien sobre video desde cero. Two-stream tenía cota dura por su dependencia del flujo óptico. La pregunta natural: ¿se puede aprovechar el progreso enorme de CNN-2D en ImageNet y transferirlo a video?

### Idea clave

**Inflar arquitecturas 2D preentrenadas a 3D.** I3D (Carreira & Zisserman, DeepMind, 2017) toma una CNN-2D entrenada en ImageNet (típicamente Inception-V1) y "infla" cada filtro 2D de tamaño $k \times k$ a un filtro 3D $k \times k \times k$, replicando los pesos a lo largo de la dimensión temporal. La inicialización transfiere la representación visual aprendida en imágenes; el fine-tuning sobre el nuevo dataset Kinetics-400 ajusta la dimensión temporal. I3D destronó a iDT y two-stream simultáneamente.

R(2+1)D (Tran et al., FAIR, 2018) propuso una factorización: cada bloque hace una convolución 2D (espacial) seguida de una 1D (temporal). Mejor accuracy con menos parámetros — y demuestra que separar espacio y tiempo es pedagógicamente útil incluso para el modelo.

SlowFast (Feichtenhofer et al., FAIR, 2019) llevó la idea a su versión más bonita: **dos pathways paralelas** inspiradas en la división Magnocellular/Parvocellular del sistema visual humano. Una "slow" a baja frame rate captura apariencia con muchos canales; una "fast" a alta frame rate captura movimiento con pocos canales. Estado del arte en Kinetics y AVA durante 2019-2020.

### Qué la destronó

Las 3D-CNN tenían un sesgo inductivo de localidad espacial y temporal — un kernel 3×3×3 solo ve un vecindario inmediato. Para acciones largas o relaciones espaciales globales, esto era un cuello de botella. La pregunta abierta de finales de 2010s era si la atención (ya dominante en NLP y avanzando en visión con ViT) podía superar a las 3D-CNN en video.

## Era 4 — Video Transformers (2021-2022)

### Problema heredado

ViT había mostrado que un Transformer puro sobre parches de imagen, con suficiente data, supera a CNNs. La extensión natural a video no era trivial: aplicar ViT cuadro por cuadro pierde temporal; aplicar atención sobre todos los parches de todos los frames escala cuadráticamente — un clip de 16 frames a 14×14 parches son ~3000 tokens, vs ~200 en una imagen estática.

### Idea clave

**Atención factorizada espacio-tiempo.** TimeSformer (Bertasius et al., FAIR, 2021) tokeniza cada frame como ViT y aplica dos atenciones por bloque: una espacial (dentro de cada frame), una temporal (a través de frames en la misma posición espacial). Cuadrático $O(T \cdot S^2 + S \cdot T^2)$ en lugar de $O((TS)^2)$. Superó a 3D-CNN profundas con menos cómputo.

ViViT (Arnab et al., Google, 2021) sistematizó el espacio: cuatro variantes de factorización (joint, factorized encoder, factorized self-attention, factorized dot-product). MViT (Fan et al., FAIR, 2021) agregó jerarquía estilo CNN — resolución decreciente, canales crecientes — para eficiencia. Video Swin (2022) extendió Swin a 3D con ventanas espacio-temporales desplazadas, recuperando sesgo inductivo local.

### Qué la destronó

Para 2022 los Video Transformers dominaban benchmarks de comprensión (Kinetics-600/700, Something-Something-V2). Pero la frontera del campo se movió hacia algo que ningún Video Transformer puro podía hacer: **generar** video coherente desde texto. Eso requería arquitecturas distintas — modelos de difusión.

## Era 5 — Generación + foundation (2022-presente)

### Problema heredado

La comprensión de video estaba madura; la generación apenas comenzaba. Modelos texto-a-imagen (DALL·E 2, Stable Diffusion, Imagen) habían explotado en 2022. La pregunta abierta: ¿se podía generar video coherente — apariencia + movimiento + identidad consistente — y a qué duración?

### Idea clave

**Diffusion + escala + tokens latentes.** Make-A-Video (Meta, 2022) e Imagen Video (Google, 2022) combinaron modelos texto-a-imagen preentrenados con módulos temporales aprendidos sobre video sin etiquetas, generando clips de pocos segundos. Stable Video Diffusion (2023) democratizó la idea con open weights.

El salto cualitativo llegó con **Sora** (OpenAI, 2024): generación de hasta 60 segundos con coherencia temporal extendida, física aproximada y resolución alta. Internamente, Sora opera sobre **tokens de video latentes** (parches espacio-temporales comprimidos), aplicando un Transformer de difusión que escala con cómputo de forma similar a los LLMs. Veo (Google DeepMind, 2024) ofreció una alternativa con foco en 4K y control de cámara. Kling (Kuaishou, 2024) demostró que China alcanzó paridad rápidamente. Runway Gen-3 llevó la generación a producción para creadores y estudios.

### Qué viene

Las apuestas activas: **coherencia física genuina** (más allá de Sora — conservación de masa, identidad estable, causalidad correcta), **video largo generativo** (5+ minutos coherentes con narrativa), **edición de video por prompt** (modificar contenido existente, no solo generar nuevo), **vision-language-action** para robótica (RT-2, π0 — pendientes en la Ola 5 de Dominios), y **modelos eficientes** para edge/móvil. La pregunta abierta de 2025: si frontier LLMs absorben video nativamente como entrada (GPT-4o, Gemini 2.5) y la generación converge en arquitecturas tipo Sora, ¿queda "video" como dominio aislado o se diluye en multimodal general?
```

**Step 2: Verify build**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify
```

**Step 3: Curl-based validation**

```bash
hugo server -D --port 1313 > /tmp/hugo-task2-ola4.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/video/

curl -s -o /tmp/video.html "$URL"

# 5 era H2s
grep -c '<h2[^>]*>Era ' /tmp/video.html  # 5

# Subsections (each H3 generates 2 anchor refs)
grep -c "Problema heredado" /tmp/video.html  # 10
grep -c "Idea clave" /tmp/video.html  # 10
grep -c "Qué la destronó" /tmp/video.html  # 8
grep -c "Qué viene" /tmp/video.html  # 2

# KaTeX inline math
grep -E 'k \\\\times k|k\\times k' /tmp/video.html | head -1
grep -E 'O\(T' /tmp/video.html | head -1

# Specific phrases
grep "Wang & Schmid, 2011" /tmp/video.html | head -1
grep "Simonyan & Zisserman, 2014" /tmp/video.html | head -1
grep "Carreira & Zisserman, DeepMind, 2017" /tmp/video.html | head -1
grep "Bertasius et al., FAIR, 2021" /tmp/video.html | head -1
grep "OpenAI, 2024" /tmp/video.html | head -1

pkill -f "hugo server" || true
sleep 1
```

**Step 4: Commit**

```bash
git add site/content/dominios/video/_index.md
git commit -m "feat(dominios/video): eras explicadas (5 subsecciones narrativas)"
```

NO Co-Authored-By trailer.

---

## Task 3: Video — SOTA + casos de uso + qué viene + recursos

**Files:**
- Modify: `site/content/dominios/video/_index.md` (apend al final)

**Step 1: Verificar fundamentos y papers para los recursos**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
ls site/content/fundamentos/{redes-convolucionales,vision-transformer,self-attention,transformer}.md 2>&1
ls site/content/papers/{alexnet-krizhevsky-2012,resnet-he-2015,vit-dosovitskiy-2021,attention-is-all-you-need-vaswani-2017}.md 2>&1
```

Si alguno falta, eliminar el bullet y reportar.

**Step 2: Apender este contenido al final**

```markdown

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier video (2024-2025).** La comprensión de video se integra a los frontier LLMs como capacidad nativa; la generación texto-a-video alcanza calidad de producción y duraciones útiles para creadores.

- **Sora 2** — OpenAI. Generación de video con coherencia temporal extendida (60s+), física aproximada y resolución alta.
- **Veo 2** — Google DeepMind. Video en 4K con prompts complejos y control de cámara cinematográfico.
- **Kling v2** — Kuaishou. Líder en China; calidad competitiva con frontier occidental.
- **Runway Gen-4 / Gen-3 Alpha** — producción para creadores; integración en pipelines de Hollywood.
- **Stable Video Diffusion 3** — Stability AI. Open weights para video corto, base de mucha experimentación.
- **Pika 2.0** — generación con control de movimiento, ediciones y transiciones.
- **Comprensión nativa**: GPT-4o, Gemini 2.5 y Claude analizan video largo como una modalidad más, sin pipeline.

{{< /callout >}}

## Casos de uso reales

- **Generación de video corto**: Sora, Runway, Pika, Kling — marketing, redes sociales, previsualización publicitaria.
- **VFX y postproducción**: integración de generación en pipelines de Hollywood (Wonder Studio, Runway, Adobe Firefly Video).
- **Acción y video clasificación**: Kinetics, Something-Something — tracking de actividad para deportes, vigilancia, seguridad.
- **Conducción autónoma**: análisis de video en tiempo real para predicción de trayectorias y planning (Tesla FSD, Waymo).
- **Análisis deportivo**: tracking de jugadores, generación automática de highlights, análisis táctico.
- **Comprensión de video largo**: resumen automático de reuniones, podcasts, lectures, contenido educativo.
- **Robótica e interacción**: video como entrada para Vision-Language-Action models (RT-2, π0).
- **Detección de eventos críticos**: vigilancia en aeropuertos, hospitales, fábricas — flag automático de incidentes.

## Qué viene

Las apuestas activas en video: **coherencia física genuina** (más allá de Sora — conservación de masa, identidad estable a través del clip, causalidad correcta), **video largo generativo** (5+ minutos con narrativa coherente), **edición de video por prompt** (modificar contenido existente, no solo generar nuevo — el equivalente de Photoshop generativo para video), **vision-language-action** para robótica (RT-2, π0, OpenVLA — cubierto en profundidad en la Ola 5 de Dominios, dominio Robótica/RL), **modelos eficientes** para edge/móvil que generen en dispositivo, y **detección de deepfakes de video** como contramedida industrial. La pregunta abierta de 2025: si los frontier LLMs absorben video nativamente como entrada y la generación converge en arquitecturas tipo Sora, ¿queda "video" como dominio aislado o se diluye en modelado multimodal general?

## Recursos relacionados

**Fundamentos (predecesores conceptuales):**
- [Redes convolucionales](/fundamentos/redes-convolucionales) — base de C3D, I3D, R(2+1)D, SlowFast (toda la era 2-3 de video).
- [Vision Transformer](/fundamentos/vision-transformer) — ViT es el ancestro directo de TimeSformer, ViViT, Video Swin.
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer) — la arquitectura sobre la que corren los Video Transformers y los modelos generativos modernos.

**Papers (predecesores adyacentes):**
- [AlexNet (Krizhevsky 2012)](/papers/alexnet-krizhevsky-2012) — la CNN-2D que motivó toda la generación de modelos 2014+.
- [ResNet (He 2015)](/papers/resnet-he-2015) — arquitectura sobre la que se inflaron muchos modelos 3D.
- [ViT (Dosovitskiy 2021)](/papers/vit-dosovitskiy-2021) — la base directa de TimeSformer y ViViT.
- [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017) — la arquitectura Transformer que vertebra la era 4-5.

**Dominios relacionados:**
- [Visión](/dominios/vision) — donde nacieron las CNN y ViT, transferidos a video.
- [Multimodal](/dominios/multimodal) — donde video se combina con texto (text-to-video) en frontier LLMs.
- [Texto / NLP](/dominios/texto) — donde nació la atención y el Transformer, transferidos a video post-2021.

---

*Última actualización: 2026-05-05.*
```

**Step 3: Verify build with curl**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo --gc --minify

hugo server -D --port 1313 > /tmp/hugo-task3-ola4.log 2>&1 &
sleep 3

URL=http://localhost:1313/neural-network-knowledge/dominios/video/

curl -s -o /tmp/video.html "$URL"

# Sections
grep -c "Estado del arte hoy" /tmp/video.html  # 3
grep -c "Casos de uso reales" /tmp/video.html  # 3
grep -c "Recursos relacionados" /tmp/video.html  # 3

# Callout
grep -c "callout" /tmp/video.html | head -1

# SOTA bullets
grep "Sora 2" /tmp/video.html | head -1
grep "Veo 2" /tmp/video.html | head -1
grep "Kling v2" /tmp/video.html | head -1
grep "Runway Gen-4" /tmp/video.html | head -1

# Resource links resolve
grep -oE 'href="[^"]*fundamentos/redes-convolucionales"' /tmp/video.html | head -1
grep -oE 'href="[^"]*fundamentos/vision-transformer"' /tmp/video.html | head -1
grep -oE 'href="[^"]*papers/vit-dosovitskiy-2021"' /tmp/video.html | head -1
grep -oE 'href="[^"]*dominios/vision"' /tmp/video.html | head -1
grep -oE 'href="[^"]*dominios/multimodal"' /tmp/video.html | head -1

# Last update
grep "Última actualización: 2026-05-05" /tmp/video.html | head -1

pkill -f "hugo server" || true
sleep 1
```

**Step 4: Commit**

```bash
git add site/content/dominios/video/_index.md
git commit -m "feat(dominios/video): SOTA, casos de uso, que viene y recursos"
```

NO Co-Authored-By trailer.

---

## Task 4: Verificación final, build de producción y push

**Files:** ninguno nuevo.

**Step 1: Confirmar branch**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git branch --show-current  # must be feat/dominios-ola-4
git log --oneline feat/dominios-ola-4 ^main
```

**Step 2: Build limpio de producción**

```bash
rm -rf site/public site/resources/_gen 2>/dev/null
cd site && hugo --gc --minify
```

Expected: build sin errores ni warnings nuevos. ~298 pages.

**Step 3: FlexSearch indexa la página nueva**

```bash
python3 -c "
import json
d = json.load(open('public/es.search-data.json'))
keys = [k for k in d.keys() if 'video' in k.lower()]
print('Video entries:', len(keys))
for k in keys:
    title = d[k].get('title', '?') if isinstance(d[k], dict) else '?'
    print(' -', k, '|', title)
"
grep -c "Sora\|Two-Stream\|I3D\|TimeSformer\|Veo" public/es.search-data.json
```

Expected: la página `/neural-network-knowledge/dominios/video/` aparece. Términos clave presentes.

**Step 4: Verificar que las páginas previas y stubs siguen funcionando**

```bash
ls public/dominios/{texto,vision,multimodal,audio,video,robotica,estructurados}/index.html
```

Expected: todas existen.

**Step 5: Push y abrir PR**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
git push -u origin feat/dominios-ola-4
```

```bash
gh pr create --base main --head feat/dominios-ola-4 --title "feat(dominios): Ola 4 — Video" --body "$(cat <<'EOF'
## Summary

Página completa para el dominio **Video** de la sección Dominios. Patrón idéntico al de Audio (Ola 3).

- **Línea de tiempo de 5 eras**: pre-deep / handcrafted (2003-2013) → two-stream y 3D-CNN tempranas (2014-2015) → 3D-CNN profundas (2017-2019) → Video Transformers (2021-2022) → generación + foundation (2022-presente).
- **19 hitos** distribuidos 3+3+3+4+6: 0 deep, 0 covered, 19 minimal. Sin material específico de video en el site, todos descripción inline (qué + por qué importó).
- **Eras explicadas** (5 subsecciones con Problema heredado / Idea clave / Qué la destronó o Qué viene).
- **Estado del arte 2024-2025** (Sora 2, Veo 2, Kling, Runway Gen-4), casos de uso, qué viene, recursos enlazando a fundamentos adyacentes (CNN, ViT, Transformer).

Diseño: docs/plans/2026-05-05-dominios-ola-4-design.md. Plan: docs/plans/2026-05-05-dominios-ola-4-plan.md.

Restan tras esta ola: Robótica/RL y Datos estructurados (Ola 5).

## Test plan

- [ ] cd site && hugo --gc --minify build limpio.
- [ ] Inspección visual desktop + móvil + dark mode en /dominios/video/.
- [ ] Búsqueda FlexSearch encuentra "Sora", "I3D", "TimeSformer", "Two-Stream" y lleva a la página.
- [ ] Click en links de Recursos llevan a Fundamentos / Papers existentes (no 404).
- [ ] Stubs (robotica, estructurados) siguen renderizando con su mensaje "Página en construcción".
- [ ] Páginas completas previas (texto, vision, multimodal, audio) renderizan correctamente.
EOF
)"
```

Reportar la URL de la PR creada.

**No commit en este task** — solo verificación, push y PR.

---

## Definition of Done — Ola 4 (Video)

- [ ] `/dominios/video/` página completa: 5 eras + 19 hitos + 5 era subsections + SOTA + casos + qué viene + recursos.
- [ ] Mínimo 800 palabras de prosa narrativa fuera de la timeline.
- [ ] Todos los `link` en recursos resuelven a archivos existentes.
- [ ] `hugo --gc --minify` build limpio.
- [ ] FlexSearch indexa la página nueva.
- [ ] Stubs (robotica, estructurados) intactos.
- [ ] Páginas completas (texto, vision, multimodal, audio) intactas.
- [ ] Branch `feat/dominios-ola-4` pusheada y PR abierta contra `main`.
- [ ] Commits sin Co-Authored-By trailer.

## Riesgos durante implementación

1. **Datos puntuales (años, autores) sin verificar** — Code reviewer subagent debe validar especialmente fechas de Era 5 (Sora, Veo, Kling), que son recientes y públicas.
2. **Solapamiento con Era 5 de Visión y Multimodal** (Sora, Stable Video Diffusion) — intencional, cada página los enmarca desde su ángulo.
3. **El usuario puede mergear/cambiar ramas durante la sesión** — verificar `git branch --show-current` tras cada subagent.
4. **KaTeX `$k \times k$` y `$O(T \cdot S^2)$`** en Era 4 — verificar en curl checks de Task 2 que renderizan; ajustar escapado si falla.
