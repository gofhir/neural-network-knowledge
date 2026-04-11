# Diseno: Sitio Hugo para Diplomado IA UC

**Fecha:** 2026-04-10
**Estado:** Aprobado

---

## Objetivo

Crear un sitio web estatico con Hugo + Hextra para documentar el Diplomado de Inteligencia Artificial de la UC. El sitio organiza apuntes de clase, analisis de papers, laboratorios (notebooks), formulas matematicas y ejemplos de codigo en una estructura navegable con busqueda integrada.

## Decisiones de diseno

| Aspecto | Decision | Razon |
|---|---|---|
| Framework | Hugo + Hextra v0.12.0 | Ya conocido por el autor (sitio GoFHIR), rapido, ligero |
| Idioma | Solo espanol | Todo el contenido esta en espanol |
| Formulas | KaTeX (ecuaciones clave) + code blocks (pseudocodigo) | Balance entre presentacion y esfuerzo de conversion |
| Notebooks | HTML embebido (nbconvert) + enlace descarga .ipynb | Fidelidad visual completa con opcion de descarga |
| Papers | Pagina de analisis + PDF descargable | Analisis propio + acceso al original |
| Busqueda | Flexsearch (integrado en Hextra) | Sin dependencias adicionales |
| Deploy | GitHub Pages con GitHub Actions | Gratis, automatico |
| Colores | Azul UC (#005a9c) como primary | Coherente con branding del diplomado |

## Ubicacion

```
ia-uc/
├── clase_05/ ... clase_10/    (contenido fuente, sin modificar)
├── proyectos/
└── site/                      (sitio Hugo)
```

El sitio vive en `site/` dentro del repositorio existente. Los archivos fuente de las clases permanecen intactos. El contenido en `site/content/` son versiones curadas (no copias automaticas ni symlinks).

## Estructura del sitio

```
site/
├── hugo.yaml
├── package.json
├── go.mod
├── content/
│   ├── _index.md                        (landing page)
│   ├── fundamentos/                     (seccion conceptual, orden logico)
│   │   ├── _index.md
│   │   ├── representacion-datos.md
│   │   ├── arquitectura-redes.md
│   │   ├── funciones-perdida.md
│   │   ├── backpropagation.md
│   │   ├── optimizadores.md
│   │   ├── learning-rate.md
│   │   ├── regularizacion.md
│   │   ├── redes-convolucionales.md
│   │   └── historia-matematica.md
│   ├── clases/                          (referencia cronologica)
│   │   ├── _index.md
│   │   ├── clase-05/
│   │   │   ├── _index.md
│   │   │   └── teoria.md
│   │   ├── clase-06/
│   │   │   ├── _index.md
│   │   │   └── teoria.md
│   │   ├── clase-07/
│   │   │   ├── _index.md
│   │   │   ├── teoria-conceptos.md
│   │   │   └── teoria-entrenamiento.md
│   │   ├── clase-08/
│   │   │   ├── _index.md
│   │   │   └── teoria.md
│   │   ├── clase-09/
│   │   │   ├── _index.md
│   │   │   ├── teoria.md
│   │   │   ├── teoria-profunda.md
│   │   │   └── conceptos-cnn.md
│   │   └── clase-10/
│   │       ├── _index.md
│   │       ├── teoria.md
│   │       ├── profundizacion.md
│   │       └── historia-matematica.md
│   ├── laboratorios/
│   │   ├── _index.md
│   │   ├── lab-05-alexnet.md
│   │   ├── lab-07-pytorch.md
│   │   └── lab-08-entrenamiento.md
│   └── papers/
│       ├── _index.md
│       ├── dropout-srivastava-2014.md
│       ├── batch-norm-ioffe-2015.md
│       ├── adam-kingma-ba-2015.md
│       └── ...
├── static/
│   ├── icons/                           (logo light/dark)
│   ├── images/                          (GIFs, diagramas)
│   ├── papers/                          (PDFs descargables)
│   ├── notebooks/                       (archivos .ipynb)
│   └── notebooks-html/                  (HTML renderizados)
├── layouts/
│   └── shortcodes/
│       ├── math-formula.html
│       ├── paper-card.html
│       ├── notebook-viewer.html
│       ├── optimizer-comparison.html
│       └── concept-alert.html
├── assets/
│   └── css/custom.css
└── data/
    └── papers.yaml
```

## Navegacion

```
Menu principal:
  Fundamentos  -->  /fundamentos/
  Clases       -->  /clases/
  Laboratorios -->  /laboratorios/
  Papers       -->  /papers/
  Buscar       -->  (flexsearch)
  GitHub       -->  (link externo)
```

La seccion Fundamentos esta ordenada por logica pedagogica (no cronologica). La seccion Clases mantiene el orden cronologico del diplomado como referencia.

## Shortcodes personalizados

### math-formula

Renderiza ecuacion KaTeX centrada con titulo y caja destacada.

```
{{< math-formula title="Regla de actualizacion de Adam" >}}
w_t = w_{t-1} - \eta \frac{\hat{r}_t}{\sqrt{\hat{v}_t} + \epsilon}
{{< /math-formula >}}
```

### paper-card

Tarjeta visual de paper con metadatos y boton de descarga.

```
{{< paper-card
    title="Adam: A Method for Stochastic Optimization"
    authors="Kingma & Ba"
    year="2015"
    venue="ICLR 2015"
    pdf="/papers/adam-kingma-2015.pdf"
    arxiv="1412.6980" >}}
Resumen breve del paper aqui.
{{< /paper-card >}}
```

### notebook-viewer

Embebe notebook HTML renderizado con boton de descarga.

```
{{< notebook-viewer
    src="/notebooks-html/lab05.html"
    download="/notebooks/Lab_5_CNNs.ipynb"
    title="Laboratorio 5 - AlexNet y CNNs" >}}
```

### concept-alert

Caja destacada para conceptos clave.

```
{{< concept-alert type="clave" >}}
El gradiente indica la direccion; el learning rate controla el tamano del paso.
{{< /concept-alert >}}
```

Tipos: `clave` (azul), `recordar` (verde), `advertencia` (amarillo).

## Configuracion Hugo

```yaml
baseURL: ""  # Configurar al hacer deploy
title: "Diplomado IA UC"
defaultContentLanguage: es
enableRobotsTXT: true
enableGitInfo: true

module:
  imports:
    - path: github.com/imfing/hextra

params:
  search:
    enable: true
    type: flexsearch
  navbar:
    displayTitle: true
    displayLogo: true
  theme:
    default: system
    displayToggle: true
  math: true  # KaTeX global

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

## Flujo de contenido

1. Los archivos fuente viven en `clase_XX/*.md` (sin modificar)
2. El contenido curado vive en `site/content/` (adaptado para Hugo con frontmatter, KaTeX, shortcodes)
3. Los notebooks se convierten con `jupyter nbconvert --to html` y se copian a `static/notebooks-html/`
4. Los PDFs de papers se copian a `static/papers/`
5. `hugo --minify` genera el sitio en `site/public/`
6. GitHub Actions despliega a GitHub Pages

## Fases de implementacion

1. **Scaffold** -- Crear estructura Hugo, configuracion, theme, shortcodes, CSS
2. **Contenido clase 10** -- Migrar los 3 documentos de la clase 10 como contenido inicial
3. **Contenido clases 5-9** -- Migrar el resto de clases
4. **Laboratorios** -- Convertir notebooks y crear paginas
5. **Papers** -- Crear paginas de papers con PDFs
6. **Fundamentos** -- Crear seccion curada con contenido reorganizado
7. **Landing page** -- Pagina de inicio con resumen del diplomado
8. **Deploy** -- Configurar GitHub Pages
