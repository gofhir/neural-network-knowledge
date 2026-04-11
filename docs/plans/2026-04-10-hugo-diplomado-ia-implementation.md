# Sitio Hugo Diplomado IA UC - Plan de Implementacion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Crear un sitio Hugo con Hextra para documentar el Diplomado IA UC, con navegacion mixta (fundamentos + clases), formulas KaTeX, notebooks embebidos y papers descargables.

**Architecture:** Sitio Hugo con theme Hextra (modulo Go), contenido curado en markdown con shortcodes personalizados para formulas, papers y notebooks. Assets estaticos (PDFs, GIFs, notebooks) servidos desde `/static/`. Deploy a GitHub Pages.

**Tech Stack:** Hugo v0.155+extended, Hextra v0.12.0, KaTeX, flexsearch, GitHub Actions

**Design doc:** `docs/plans/2026-04-10-hugo-diplomado-ia-site-design.md`

---

## Fase 1: Scaffold del sitio Hugo (funcional al terminar)

### Task 1: Inicializar proyecto Hugo

**Files:**
- Create: `site/hugo.yaml`
- Create: `site/go.mod`
- Create: `site/package.json`
- Create: `site/assets/css/custom.css`

**Step 1: Crear directorio e inicializar modulo Go**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc
mkdir -p site
cd site
hugo mod init github.com/robertoaraneda/diplomado-ia-uc/site
```

**Step 2: Crear hugo.yaml**

```yaml
baseURL: ""
title: "Diplomado IA UC"
defaultContentLanguage: es
enableRobotsTXT: true
enableGitInfo: false

module:
  hugoVersion:
    extended: true
    min: "0.112.0"
  imports:
    - path: github.com/imfing/hextra

menu:
  main:
    - name: Fundamentos
      pageRef: /fundamentos
      weight: 10
    - name: Clases
      pageRef: /clases
      weight: 20
    - name: Laboratorios
      pageRef: /laboratorios
      weight: 30
    - name: Papers
      pageRef: /papers
      weight: 40
    - name: Buscar
      weight: 50
      params:
        type: search
    - name: GitHub
      url: https://github.com/robertoaraneda/diplomado-ia-uc
      weight: 60
      params:
        icon: github

params:
  search:
    enable: true
    type: flexsearch
    flexsearch:
      index: content
  navbar:
    displayTitle: true
    displayLogo: true
    logo:
      path: icons/logo.svg
      dark: icons/logo-dark.svg
  theme:
    default: system
    displayToggle: true
  footer:
    displayCopyright: true
    displayPoweredBy: false

markup:
  goldmark:
    parser:
      attribute:
        block: true
    renderer:
      unsafe: true
  highlight:
    noClasses: false
```

**Step 3: Crear package.json**

```json
{
  "name": "diplomado-ia-uc",
  "version": "1.0.0",
  "private": true,
  "description": "Diplomado Inteligencia Artificial - UC",
  "scripts": {
    "serve": "hugo server --buildDrafts --disableFastRender",
    "build": "hugo --minify",
    "clean": "rm -rf public/ resources/_gen/"
  }
}
```

**Step 4: Crear custom.css**

```css
:root {
  /* UC blue: #005a9c → hsl(209, 100%, 31%) */
  --primary-hue: 209deg;
  --primary-saturation: 100%;
  --primary-lightness: 31%;
}

/* KaTeX formula boxes */
.math-formula-box {
  background: var(--card-bg, #f8fafc);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin: 1.5rem 0;
  text-align: center;
}

.math-formula-box .math-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-secondary, #64748b);
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Paper cards */
.paper-card {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 0.5rem;
  padding: 1.25rem;
  margin: 1rem 0;
  transition: box-shadow 0.2s;
}

.paper-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.paper-card .paper-meta {
  font-size: 0.8rem;
  color: var(--text-secondary, #64748b);
}

/* Concept alerts */
.concept-alert {
  border-radius: 0.5rem;
  border-left: 4px solid;
  padding: 1rem 1.25rem;
  margin: 1rem 0;
}

.concept-alert-clave {
  background: #eff6ff;
  border-color: #3b82f6;
}

.concept-alert-recordar {
  background: #f0fdf4;
  border-color: #22c55e;
}

.concept-alert-advertencia {
  background: #fffbeb;
  border-color: #f59e0b;
}

:is(.dark) .concept-alert-clave {
  background: rgba(59, 130, 246, 0.1);
}

:is(.dark) .concept-alert-recordar {
  background: rgba(34, 197, 94, 0.1);
}

:is(.dark) .concept-alert-advertencia {
  background: rgba(245, 158, 11, 0.1);
}

/* Notebook viewer */
.notebook-viewer {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 0.5rem;
  overflow: hidden;
  margin: 1rem 0;
}

.notebook-viewer iframe {
  width: 100%;
  min-height: 600px;
  border: none;
}

.notebook-viewer .notebook-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: var(--card-bg, #f8fafc);
  border-bottom: 1px solid var(--border-color, #e2e8f0);
}
```

**Step 5: Descargar el theme**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
hugo mod get github.com/imfing/hextra@v0.12.0
hugo mod vendor
```

**Step 6: Verificar que Hugo compila**

```bash
hugo --minify 2>&1 | tail -5
```

Expected: Build succeeds (puede haber warnings por falta de contenido, esta OK).

---

### Task 2: Crear shortcodes personalizados

**Files:**
- Create: `site/layouts/shortcodes/math-formula.html`
- Create: `site/layouts/shortcodes/paper-card.html`
- Create: `site/layouts/shortcodes/notebook-viewer.html`
- Create: `site/layouts/shortcodes/concept-alert.html`

**Step 1: Crear math-formula shortcode**

`site/layouts/shortcodes/math-formula.html`:

```html
{{- $title := .Get "title" | default "" -}}
<div class="math-formula-box">
  {{- if $title }}
  <div class="math-title">{{ $title }}</div>
  {{- end }}
  <div>$$ {{ .Inner }} $$</div>
</div>
```

**Step 2: Crear paper-card shortcode**

`site/layouts/shortcodes/paper-card.html`:

```html
{{- $title := .Get "title" -}}
{{- $authors := .Get "authors" -}}
{{- $year := .Get "year" -}}
{{- $venue := .Get "venue" | default "" -}}
{{- $pdf := .Get "pdf" | default "" -}}
{{- $arxiv := .Get "arxiv" | default "" -}}
<div class="paper-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
    <div>
      <strong style="font-size:1.05rem;">{{ $title }}</strong>
      <div class="paper-meta" style="margin-top:0.25rem;">
        {{ $authors }} ({{ $year }}){{ with $venue }} &mdash; {{ . }}{{ end }}
      </div>
      {{- with .Inner }}
      <div style="margin-top:0.5rem; font-size:0.9rem;">{{ . | markdownify }}</div>
      {{- end }}
    </div>
    <div style="display:flex; gap:0.5rem; flex-shrink:0;">
      {{- with $arxiv }}
      <a href="https://arxiv.org/abs/{{ . }}" target="_blank" rel="noopener" style="font-size:0.8rem; padding:0.25rem 0.75rem; border:1px solid #e2e8f0; border-radius:0.375rem; text-decoration:none; white-space:nowrap;">arXiv</a>
      {{- end }}
      {{- with $pdf }}
      <a href="{{ . }}" target="_blank" rel="noopener" style="font-size:0.8rem; padding:0.25rem 0.75rem; background:#005a9c; color:white; border-radius:0.375rem; text-decoration:none; white-space:nowrap;">PDF</a>
      {{- end }}
    </div>
  </div>
</div>
```

**Step 3: Crear notebook-viewer shortcode**

`site/layouts/shortcodes/notebook-viewer.html`:

```html
{{- $src := .Get "src" -}}
{{- $download := .Get "download" | default "" -}}
{{- $title := .Get "title" | default "Notebook" -}}
<div class="notebook-viewer">
  <div class="notebook-toolbar">
    <span style="font-weight:600; font-size:0.9rem;">{{ $title }}</span>
    <div style="display:flex; gap:0.5rem;">
      {{- with $download }}
      <a href="{{ . }}" download style="font-size:0.8rem; padding:0.25rem 0.75rem; background:#005a9c; color:white; border-radius:0.375rem; text-decoration:none;">Descargar .ipynb</a>
      {{- end }}
    </div>
  </div>
  <iframe src="{{ $src }}" loading="lazy"></iframe>
</div>
```

**Step 4: Crear concept-alert shortcode**

`site/layouts/shortcodes/concept-alert.html`:

```html
{{- $type := .Get "type" | default "clave" -}}
{{- $icon := "💡" -}}
{{- if eq $type "recordar" }}{{ $icon = "📌" }}{{ end -}}
{{- if eq $type "advertencia" }}{{ $icon = "⚠️" }}{{ end -}}
<div class="concept-alert concept-alert-{{ $type }}">
  <span>{{ $icon }}</span> {{ .Inner | markdownify }}
</div>
```

---

### Task 3: Crear landing page y estructura de secciones

**Files:**
- Create: `site/content/_index.md`
- Create: `site/content/fundamentos/_index.md`
- Create: `site/content/clases/_index.md`
- Create: `site/content/laboratorios/_index.md`
- Create: `site/content/papers/_index.md`

**Step 1: Crear landing page**

`site/content/_index.md`:

```markdown
---
title: Diplomado IA UC
layout: hextra-home
---

<div class="hx-mt-6 hx-mb-6">
{{< hextra/hero-headline >}}
  Diplomado Inteligencia Artificial
{{< /hextra/hero-headline >}}
</div>

<div class="hx-mb-12">
{{< hextra/hero-subtitle >}}
  Apuntes, analisis y recursos del Diplomado de IA&nbsp;<br class="sm:hx-block hx-hidden" />de la Escuela de Ingenieria UC
{{< /hextra/hero-subtitle >}}
</div>

<div class="hx-mb-6">
{{< hextra/hero-button text="Empezar por Fundamentos" link="fundamentos" >}}
</div>

<div class="hx-mt-6"></div>

{{< hextra/feature-grid >}}
  {{< hextra/feature-card
    title="Fundamentos"
    subtitle="Teoria organizada por tema: backpropagation, optimizadores, CNNs, regularizacion."
    link="fundamentos"
  >}}
  {{< hextra/feature-card
    title="Clases"
    subtitle="Apuntes y analisis de cada sesion del diplomado en orden cronologico."
    link="clases"
  >}}
  {{< hextra/feature-card
    title="Laboratorios"
    subtitle="Notebooks interactivos con implementaciones en PyTorch, JAX y TensorFlow."
    link="laboratorios"
  >}}
  {{< hextra/feature-card
    title="Papers"
    subtitle="Analisis de papers fundamentales: Adam, Dropout, Batch Normalization, ResNets."
    link="papers"
  >}}
{{< /hextra/feature-grid >}}
```

**Step 2: Crear index de Fundamentos**

`site/content/fundamentos/_index.md`:

```markdown
---
title: Fundamentos
weight: 10
sidebar:
  open: true
---

Teoria de deep learning organizada por tema, en orden logico de aprendizaje. El contenido esta curado y complementado con investigacion adicional mas alla de lo cubierto en las clases.
```

**Step 3: Crear index de Clases**

`site/content/clases/_index.md`:

```markdown
---
title: Clases
weight: 20
sidebar:
  open: true
---

Apuntes y analisis de cada sesion del diplomado, en orden cronologico.
```

**Step 4: Crear index de Laboratorios**

`site/content/laboratorios/_index.md`:

```markdown
---
title: Laboratorios
weight: 30
---

Notebooks con implementaciones practicas. Cada laboratorio incluye el notebook renderizado y un enlace para descargar el archivo `.ipynb` original.
```

**Step 5: Crear index de Papers**

`site/content/papers/_index.md`:

```markdown
---
title: Papers
weight: 40
---

Analisis de papers fundamentales referenciados en el diplomado, con enlaces a los PDFs originales.
```

**Step 6: Crear logo placeholder**

```bash
mkdir -p site/static/icons
# SVG minimo como placeholder
cat > site/static/icons/logo.svg << 'SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><text x="0" y="20" font-size="20" font-family="serif" fill="#005a9c">IA</text></svg>
SVG
cp site/static/icons/logo.svg site/static/icons/logo-dark.svg
```

**Step 7: Verificar que el sitio compila y se sirve**

```bash
cd /Users/robertoaraneda/projects/personal/courses/ia-uc/site
npm run serve
```

Expected: Sitio corriendo en localhost:1313 con landing page, menu de 4 secciones, dark mode toggle, y busqueda.

**Step 8: Commit**

```bash
git add site/
git commit -m "feat(site): scaffold Hugo site with Hextra, shortcodes, and section structure"
```

---

## Fase 2: Contenido de Clase 10 (primera clase completa)

### Task 4: Copiar assets estaticos de clase 10

**Files:**
- Copy GIFs to: `site/static/images/clase-10/`
- Copy PDF to: `site/static/papers/`

**Step 1: Copiar GIFs y PDF de la clase 10**

```bash
mkdir -p site/static/images/clase-10
mkdir -p site/static/papers

cp clase_10/material/Clase\ Teorica/*.gif site/static/images/clase-10/
cp "clase_10/material/Clase Teorica/Clase 10 - Algoritmos de optimizacion y learning rate.pdf" site/static/papers/clase10-optimizacion-learning-rate.pdf
```

---

### Task 5: Crear contenido de Clase 10 en la seccion Clases

**Files:**
- Create: `site/content/clases/clase-10/_index.md`
- Create: `site/content/clases/clase-10/teoria.md`
- Create: `site/content/clases/clase-10/profundizacion.md`
- Create: `site/content/clases/clase-10/historia-matematica.md`

**Step 1: Crear index de clase 10**

`site/content/clases/clase-10/_index.md`:

```markdown
---
title: "Clase 10 - Optimizacion y Learning Rate"
weight: 60
sidebar:
  open: true
---

**Profesora:** Francisca Cattan Castillo
**Fecha:** 2026-04-10

Algoritmos de optimizacion (GD, SGD, Momentum, Nesterov, AdaGrad, Adam), learning rate scheduling, y early stopping.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Contenido de las 53 diapositivas" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Loss functions, backprop, LR en detalle" icon="beaker" >}}
  {{< card link="historia-matematica" title="Historia Matematica" subtitle="De Cauchy (1847) a Adam (2015)" icon="book-open" >}}
{{< /cards >}}
```

**Step 2: Crear teoria.md**

Copiar y adaptar el contenido de `clase_10/clase_10_teoria_optimizacion_y_learning_rate.md` con frontmatter Hugo, convertir las formulas principales a KaTeX, y agregar las imagenes GIF.

`site/content/clases/clase-10/teoria.md`:

Frontmatter:
```yaml
---
title: "Teoria: Algoritmos de Optimizacion"
weight: 10
math: true
---
```

Cuerpo: copiar el contenido del archivo original, ajustando:
- Las formulas principales (regla de actualizacion, SGD, momentum, Nesterov, AdaGrad, Adam) a sintaxis KaTeX con `$$...$$`
- Tablas y pseudocodigo se mantienen como code blocks
- Agregar imagenes GIF donde corresponda: `![SGD Momentum](/images/clase-10/sgd_momentum.gif)`
- Agregar shortcodes `{{< concept-alert >}}` para conceptos clave

**Step 3: Crear profundizacion.md**

Misma logica: copiar de `clase_10/clase_10_profundizacion_optimizacion_backprop_lr.md` con frontmatter:

```yaml
---
title: "Profundizacion: Loss, Backprop, LR"
weight: 20
math: true
---
```

Convertir formulas clave a KaTeX. Las derivaciones largas se mantienen en code blocks.

**Step 4: Crear historia-matematica.md**

Copiar de `clase_10/clase_10_raices_matematicas_e_historia.md`:

```yaml
---
title: "Historia Matematica de la Optimizacion"
weight: 30
math: true
---
```

**Step 5: Verificar render local**

```bash
cd site && npm run serve
```

Navegar a `/clases/clase-10/` y verificar que las 3 paginas renderizan correctamente con formulas KaTeX, GIFs, y tablas.

**Step 6: Commit**

```bash
git add site/content/clases/clase-10/ site/static/images/clase-10/ site/static/papers/clase10-*.pdf
git commit -m "feat(site): add clase 10 content - optimization and learning rate"
```

---

## Fase 3: Contenido de Clases 5-9

### Task 6: Clase 05 - Redes Convolucionales

**Files:**
- Create: `site/content/clases/clase-05/_index.md`
- Create: `site/content/clases/clase-05/teoria.md`

**Step 1: Crear index y contenido**

`site/content/clases/clase-05/_index.md`:
```yaml
---
title: "Clase 05 - Redes Convolucionales"
weight: 10
sidebar:
  open: true
---
```

**Step 2: Crear teoria.md** adaptando `clase_05/laboratorio5_alexnet.md` con frontmatter y KaTeX.

**Step 3: Commit**

```bash
git add site/content/clases/clase-05/
git commit -m "feat(site): add clase 05 content - convolutional networks"
```

---

### Task 7: Clase 06 - Practica

**Files:**
- Create: `site/content/clases/clase-06/_index.md`
- Create: `site/content/clases/clase-06/teoria.md`

**Step 1:** Adaptar `clase_06/practico_clase_6_teoria.md`.

**Step 2: Commit**

```bash
git add site/content/clases/clase-06/
git commit -m "feat(site): add clase 06 content"
```

---

### Task 8: Clase 07 - Tecnicas de Entrenamiento

**Files:**
- Create: `site/content/clases/clase-07/_index.md`
- Create: `site/content/clases/clase-07/conceptos.md`
- Create: `site/content/clases/clase-07/entrenamiento-pytorch.md`

**Step 1:** Adaptar `clase_07/clase7_conceptos_y_definiciones.md` y `clase_07/clase7_tecnicas_entrenamiento_pytorch.md`.

**Step 2:** Copiar papers PDF a `site/static/papers/`:
```bash
cp clase_07/papers/*.pdf site/static/papers/
```

**Step 3: Commit**

```bash
git add site/content/clases/clase-07/ site/static/papers/
git commit -m "feat(site): add clase 07 content - training techniques"
```

---

### Task 9: Clase 08 - Funciones de Perdida

**Files:**
- Create: `site/content/clases/clase-08/_index.md`
- Create: `site/content/clases/clase-08/teoria.md`

**Step 1:** Adaptar `clase_08/clase8_funciones_perdida_regularizacion_tareas_auxiliares.md`.

**Step 2: Commit**

```bash
git add site/content/clases/clase-08/
git commit -m "feat(site): add clase 08 content - loss functions"
```

---

### Task 10: Clase 09 - CNNs en Profundidad

**Files:**
- Create: `site/content/clases/clase-09/_index.md`
- Create: `site/content/clases/clase-09/teoria.md`
- Create: `site/content/clases/clase-09/profundizacion.md`
- Create: `site/content/clases/clase-09/conceptos-cnn.md`

**Step 1:** Adaptar los 5 archivos `.md` de `clase_09/`.

**Step 2:** Copiar papers:
```bash
cp clase_09/papers/*.pdf site/static/papers/
```

**Step 3: Commit**

```bash
git add site/content/clases/clase-09/ site/static/papers/
git commit -m "feat(site): add clase 09 content - CNNs in depth"
```

---

## Fase 4: Laboratorios

### Task 11: Convertir notebooks a HTML y crear paginas

**Files:**
- Create: `site/static/notebooks-html/lab05.html`
- Create: `site/static/notebooks-html/lab07.html`
- Create: `site/static/notebooks-html/lab08.html`
- Copy: `site/static/notebooks/*.ipynb`
- Create: `site/content/laboratorios/lab-05-alexnet.md`
- Create: `site/content/laboratorios/lab-07-pytorch.md`
- Create: `site/content/laboratorios/lab-08-entrenamiento.md`

**Step 1: Convertir notebooks a HTML**

```bash
mkdir -p site/static/notebooks-html site/static/notebooks

jupyter nbconvert --to html --no-input-prompt \
  "clase_05/Laboratorio_5_Redes_Convolucionales_IA_vAlumnos(2).ipynb" \
  --output-dir site/static/notebooks-html/ --output lab05.html

jupyter nbconvert --to html --no-input-prompt \
  "clase_07/Laboratorio_7_PyTorch_Diplomado_IA_vAlumnos_RAE.ipynb" \
  --output-dir site/static/notebooks-html/ --output lab07.html

jupyter nbconvert --to html --no-input-prompt \
  "clase_08/Lab_8_RAE.ipynb" \
  --output-dir site/static/notebooks-html/ --output lab08.html
```

**Step 2: Copiar notebooks originales para descarga**

```bash
cp "clase_05/Laboratorio_5_Redes_Convolucionales_IA_vAlumnos(2).ipynb" site/static/notebooks/lab05.ipynb
cp "clase_07/Laboratorio_7_PyTorch_Diplomado_IA_vAlumnos_RAE.ipynb" site/static/notebooks/lab07.ipynb
cp "clase_08/Lab_8_RAE.ipynb" site/static/notebooks/lab08.ipynb
```

**Step 3: Crear paginas de laboratorios**

`site/content/laboratorios/lab-05-alexnet.md`:
```markdown
---
title: "Lab 05 - AlexNet y CNNs"
weight: 10
---

Laboratorio practico sobre redes convolucionales y la arquitectura AlexNet.

{{< notebook-viewer
    src="/notebooks-html/lab05.html"
    download="/notebooks/lab05.ipynb"
    title="Laboratorio 5 - Redes Convolucionales" >}}
```

Crear analogamente `lab-07-pytorch.md` (weight: 20) y `lab-08-entrenamiento.md` (weight: 30).

**Step 4: Verificar render**

```bash
cd site && npm run serve
```

Navegar a `/laboratorios/` y verificar que los notebooks se renderizan y los links de descarga funcionan.

**Step 5: Commit**

```bash
git add site/content/laboratorios/ site/static/notebooks/ site/static/notebooks-html/
git commit -m "feat(site): add laboratory notebooks with viewer and download"
```

---

## Fase 5: Papers

### Task 12: Crear paginas de papers

**Files:**
- Create: `site/content/papers/dropout-srivastava-2014.md`
- Create: `site/content/papers/batch-norm-ioffe-2015.md`
- Create: `site/content/papers/adam-kingma-ba-2015.md`
- Create: `site/content/papers/vggnet-simonyan-2014.md`
- Create: `site/content/papers/googlenet-szegedy-2014.md`
- Create: `site/content/papers/resnet-he-2015.md`

**Step 1: Crear pagina de Dropout**

`site/content/papers/dropout-srivastava-2014.md`:
```markdown
---
title: "Dropout"
weight: 10
math: true
---

{{< paper-card
    title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    authors="Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov"
    year="2014"
    venue="JMLR"
    pdf="/papers/srivastava2014_dropout.pdf" >}}
Tecnica de regularizacion que desactiva neuronas aleatoriamente durante el entrenamiento.
{{< /paper-card >}}

---
```

Seguido del contenido adaptado de `clase_07/papers/analisis_dropout_srivastava2014.md`.

**Step 2:** Crear analogamente las demas paginas de papers con el contenido disponible.

**Step 3: Commit**

```bash
git add site/content/papers/ site/static/papers/
git commit -m "feat(site): add paper analysis pages with PDF downloads"
```

---

## Fase 6: Fundamentos (seccion curada)

### Task 13: Crear seccion Fundamentos

**Files:**
- Create: `site/content/fundamentos/optimizadores.md`
- Create: `site/content/fundamentos/backpropagation.md`
- Create: `site/content/fundamentos/funciones-perdida.md`
- Create: `site/content/fundamentos/learning-rate.md`
- Create: `site/content/fundamentos/regularizacion.md`
- Create: `site/content/fundamentos/redes-convolucionales.md`
- Create: `site/content/fundamentos/historia-matematica.md`
- Create: `site/content/fundamentos/representacion-datos.md`
- Create: `site/content/fundamentos/arquitectura-redes.md`

**Step 1: Crear paginas de fundamentos**

Cada pagina tiene frontmatter con `math: true`, un `weight` para el orden logico, y contenido curado que combina material de varias clases. Ejemplo:

`site/content/fundamentos/optimizadores.md`:
```yaml
---
title: "Optimizadores"
weight: 50
math: true
---
```

Contenido curado que combina:
- Teoria de clase 10 (GD, SGD, Momentum, Nesterov, AdaGrad, Adam)
- Profundizacion de clase 10 (analisis de cada optimizador)
- Historia de clase 10 (Polyak, Nesterov, Duchi, Kingma)
- Formulas clave en KaTeX
- Cross-links a `/clases/clase-10/` para referencia

Orden logico de los fundamentos:
1. representacion-datos (weight: 10)
2. arquitectura-redes (weight: 20)
3. funciones-perdida (weight: 30)
4. backpropagation (weight: 40)
5. optimizadores (weight: 50)
6. learning-rate (weight: 60)
7. regularizacion (weight: 70)
8. redes-convolucionales (weight: 80)
9. historia-matematica (weight: 90)

**Step 2: Verificar navegacion y cross-links**

```bash
cd site && npm run serve
```

Verificar que la navegacion sidebar de Fundamentos sigue el orden logico y los links entre secciones funcionan.

**Step 3: Commit**

```bash
git add site/content/fundamentos/
git commit -m "feat(site): add curated fundamentals section"
```

---

## Fase 7: Landing page y polish final

### Task 14: Crear logo SVG del sitio

**Files:**
- Modify: `site/static/icons/logo.svg`
- Modify: `site/static/icons/logo-dark.svg`

**Step 1:** Crear un SVG simple con las letras "IA" o el escudo UC estilizado, en azul UC para light mode y blanco para dark mode.

**Step 2: Commit**

```bash
git add site/static/icons/
git commit -m "feat(site): add site logos"
```

---

### Task 15: Verificacion final y build de produccion

**Step 1: Build de produccion**

```bash
cd site
npm run clean
npm run build
```

Expected: Build exitoso sin errores. Directorio `public/` generado.

**Step 2: Verificar links rotos**

```bash
hugo --minify 2>&1 | grep -i "error\|warn"
```

**Step 3: Verificar que KaTeX renderiza**

Abrir `npm run serve` y navegar a una pagina con formulas. Verificar que `$$...$$` se renderiza como ecuaciones.

**Step 4: Verificar que los notebooks embebidos cargan**

Navegar a `/laboratorios/lab-05-alexnet/` y verificar iframe + boton de descarga.

**Step 5: Commit final**

```bash
git add -A
git commit -m "feat(site): complete Hugo site for Diplomado IA UC"
```

---

## Resumen de fases

| Fase | Tasks | Resultado |
|---|---|---|
| 1 - Scaffold | 1-3 | Sitio funcional con landing, menu, shortcodes |
| 2 - Clase 10 | 4-5 | Primera clase completa con GIFs y KaTeX |
| 3 - Clases 5-9 | 6-10 | Todas las clases migradas |
| 4 - Laboratorios | 11 | 3 notebooks embebidos con descarga |
| 5 - Papers | 12 | Paginas de papers con PDFs |
| 6 - Fundamentos | 13 | Seccion curada con orden logico |
| 7 - Polish | 14-15 | Logo, verificacion, build final |
