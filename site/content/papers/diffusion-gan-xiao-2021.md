---
title: "Denoising Diffusion GANs: el trilema generativo (2021)"
weight: 339
math: true
---

{{< paper-card
    title="Tackling the Generative Learning Trilemma with Denoising Diffusion GANs"
    authors="Zhisheng Xiao, Karsten Kreis, Arash Vahdat"
    year="2021"
    venue="ICLR 2022"
    pdf="/papers/diffusion-gan-xiao-2021.pdf"
    arxiv="2112.07804" >}}
Paper de NVIDIA y la Universidad de Chicago que articula el **trilema del aprendizaje generativo**: ningún modelo profundo logra a la vez las tres propiedades que el mundo real exige —**alta calidad**, **muestreo rápido** y **buena cobertura de modos**. Las [GANs](/papers/gan-goodfellow-2014) sacrifican cobertura, los VAEs sacrifican calidad y los [modelos de difusión](/papers/ddpm-ho-2020) sacrifican velocidad. La propuesta: modelar cada paso de denoising con una **GAN condicional** en vez de una gaussiana, lo que permite dar pocos pasos grandes (**~4 en lugar de 1000**) y ser **~2000× más rápido** sin perder calidad ni diversidad. Es el paper que la [Clase 29](/clases/clase-29) cita explícitamente para construir su marco comparativo VAE / GAN / Difusión.
{{< /paper-card >}}

---

## El trilema del aprendizaje generativo

Hacia 2021 existía una rica familia de modelos generativos profundos para imágenes, audio, nubes de puntos y grafos. Pero —argumentan Xiao, Kreis y Vahdat— ninguno satisfacía simultáneamente los tres requisitos que las aplicaciones reales suelen exigir:

1. **Alta calidad** de muestras (que las imágenes generadas sean realistas).
2. **Muestreo rápido** y computacionalmente barato (edición interactiva, síntesis de voz en tiempo real).
3. **Buena cobertura de modos** y diversidad (representar fielmente toda la distribución de datos, incluidas las minorías, para reducir sesgos y los impactos sociales negativos de los modelos generativos).

Casi todos los modelos sacrifican uno de los tres por los otros dos. Los autores bautizan ese compromiso de tres vías como el **trilema del aprendizaje generativo** (*generative learning trilemma*) y lo dibujan como un triángulo (Fig. 1) en el que cada familia clásica falla en un vértice distinto:

| Familia | Calidad | Velocidad | Cobertura |
|---|---|---|---|
| **GANs** (Goodfellow 2014, BigGAN) | alta | rápida (1 paso) | **pobre** (mode collapse) |
| **VAEs / normalizing flows** | **baja** | rápida | buena (alta verosimilitud) |
| **Modelos de difusión** (DDPM, Score SDE) | alta | **lentísima** (1000–2000 pasos) | buena |

Esta tabla es exactamente la que la Clase 29 muestra para comparar VAE / GAN / Difusión sobre los ejes **velocidad / calidad / distribución**: no es una tabla cualquiera, es la operacionalización directa del trilema de este paper. Entender el paper es entender *por qué* esa tabla está construida así y *qué* casilla intenta llenar cada modelo.

El objetivo del trabajo es atacar el vértice más caro de la difusión —su lentitud— sin renunciar a las otras dos propiedades que la difusión ya tenía resueltas.

## El diagnóstico: por qué la difusión es lenta

La contribución empieza con un diagnóstico de raíz. En un modelo de difusión hay un **proceso forward** que añade ruido gaussiano a los datos en $T$ pasos:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big),$$

y un **proceso reverso** (denoising) que parte de ruido y va limpiando, modelando cada paso como una **gaussiana**:

$$p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\!\big(x_{t-1};\mu_\theta(x_t,t),\,\sigma_t^2 I\big).$$

Aquí está el punto crítico. Por la regla de Bayes, la distribución de denoising *verdadera* $q(x_{t-1}\mid x_t)$ solo toma forma gaussiana en dos situaciones: (a) en el **límite de pasos infinitesimalmente pequeños** ($\beta_t \to 0$), donde el producto de Bayes queda dominado por el término gaussiano del forward (resultado clásico de Feller, 1949); o (b) si la marginal de datos ya fuera gaussiana. La primera condición es justamente la que **obliga a usar cientos o miles de pasos**: para que cada gaussiana sea válida, cada paso debe ser minúsculo.

Cuando se intenta dar **pasos grandes** (y por tanto pocos pasos en total), la distribución de denoising verdadera deja de ser gaussiana y se vuelve **multimodal y compleja**. La intuición en imágenes: múltiples imágenes limpias plausibles pueden corresponder a una misma imagen ruidosa, así que la denoising no apunta a un único destino sino a varios. La Fig. 2 lo ilustra con datos 1D: para un paso pequeño la denoising verdadera es casi gaussiana, pero a medida que el paso crece se vuelve cada vez más multimodal. Forzar una gaussiana unimodal sobre una distribución multimodal es lo que rompe el modelo si se reducen los pasos a la fuerza.

## El remedio: una GAN condicional por paso de denoising

Si el problema es que la gaussiana no captura una distribución multimodal, la solución es reemplazarla por una distribución expresiva. Los autores eligen modelar cada paso de denoising con un **GAN condicional**, porque las GANs condicionales ya habían demostrado capturar distribuciones condicionales complejas en imagen (Mirza & Osindero 2014, pix2pix).

El proceso forward se monta igual que en la difusión estándar, pero con $T$ **pequeño** ($T \le 8$, óptimo en $T=4$) y cada paso con $\beta_t$ **grande**. El entrenamiento hace *matching* entre el generador y la denoising verdadera minimizando una divergencia adversarial por paso, mediante un **discriminador dependiente del tiempo** $D_\phi(x_{t-1}, x_t, t)$ que decide si $x_{t-1}$ es una versión denoised plausible de $x_t$.

La pieza más elegante es la **parametrización**. En lugar de que el generador prediga $x_{t-1}$ directamente, primero predice una estimación de la imagen limpia $x_0$ y luego $x_{t-1}$ se muestrea de la posterior $q(x_{t-1}\mid x_t, x_0)$ —que *siempre* tiene forma gaussiana cerrada, sin importar el tamaño de paso:

$$p_\theta(x_{t-1}\mid x_t) = \int p(z)\,q\big(x_{t-1}\mid x_t,\,x_0 = G_\theta(x_t, z, t)\big)\,dz,$$

donde $G_\theta(x_t, z, t)$ es el generador, que recibe la imagen ruidosa $x_t$ y una **variable latente $z \sim \mathcal{N}(0,I)$**. Esto hereda el sesgo inductivo de DDPM (misma estructura de red), con una diferencia crucial: en DDPM $x_0$ se predice de forma **determinista**, mientras que aquí lo produce el generador con la latente aleatoria $z$. Es precisamente esa $z$ la que vuelve **multimodal** la distribución de denoising. La ablación lo confirma de forma contundente: quitar $z$ convierte el denoising en unimodal y degrada el FID de 3.75 a 20.6.

¿Por qué no entrenar simplemente una GAN tradicional de un solo paso? Porque descomponer la generación en varios pasos condicionados en $x_t$ hace que cada paso sea **simple de modelar** (evitando la inestabilidad y el mode collapse de generar de golpe), y porque el proceso de difusión **suaviza la distribución de datos**, lo que hace al discriminador menos propenso a sobreajustar. El resultado: más estabilidad y mejor cobertura de modos que una GAN convencional.

## Resultados: ganar en los tres ejes a la vez

Sobre CIFAR-10 (generación incondicional, arquitectura NCSN++, $T=4$), los autores miden los tres vértices del trilema simultáneamente —fidelidad (FID, IS), diversidad (recall) y tiempo de muestreo (NFE y segundos en una V100):

| Modelo | FID↓ | Recall↑ | NFE↓ | Tiempo (s)↓ |
|---|---|---|---|---|
| **Denoising Diffusion GAN (T=4)** | **3.75** | **0.57** | **4** | **0.21** |
| DDPM (Ho 2020) | 3.21 | 0.57 | 1000 | 80.5 |
| Score SDE (Song 2021) | 2.20 | 0.59 | 2000 | 423.2 |
| StyleGAN2 + ADA | 2.92 | 0.49 | 1 | 0.04 |

Las lecturas clave:

- **Velocidad:** ~**2000× más rápido** que el muestreo predictor-corrector de Score SDE (0.21 s vs 423.2 s). La Fig. 4 muestra que el modelo domina a toda la difusión previa en el trade-off calidad/tiempo.
- **Calidad:** FID 3.75, competitivo con los mejores modelos; solo StyleGAN2+ADA lo supera ligeramente en calidad pura.
- **Diversidad:** las GANs tienen recall por debajo de 0.5 (pobre cobertura); el modelo logra 0.57, a la par de la difusión. Es decir, **gana en los tres ejes a la vez** —exactamente el punto del trilema.

La cobertura se confirma en pruebas dedicadas: cubre los **1000 modos** de StackedMNIST con el menor KL (0.071), superando a GANs diseñadas para cobertura (PacGAN, PresGAN). Escala a alta resolución (256×256), superando a DDPM en LSUN Church (FID 5.25 vs 7.89). Y en edición interactiva basada en trazos (SDEdit) logra un **speedup de ~1100×** (0.16 s vs 181 s por imagen), confirmando viabilidad práctica.

Las ablaciones cierran el argumento: $T=1$ (equivalente a una GAN) da recall de apenas 0.19; predecir $x_0$ y muestrear de la posterior supera por amplio margen a predecir $x_{t-1}$ o el ruido $\epsilon$ directamente.

## Limitaciones reconocidas

- **Capacidad vs número de pasos:** el modelo necesita un GAN condicional por paso, y subir $T$ más allá de 4 degrada el rendimiento salvo que se aumente la capacidad. El óptimo $T=4$ es un punto delicado, no un parámetro libremente escalable.
- **Herencia de las GANs:** sigue entrenándose con pérdida adversarial; la estabilidad de entrenamiento no es trivial aunque el condicionamiento por pasos y el suavizado la mitiguen.
- **Calidad pura aún por debajo del tope:** algunos modelos de difusión con muchos pasos y StyleGAN2+ADA logran mejor FID aislado. El aporte es el *balance* de los tres ejes, no liderar el de calidad.
- **GAN elegida empíricamente:** un VAE condicional como denoising dio resultados pobres; la superioridad del GAN es empírica, no demostrada óptima.

## Por qué importa para la Clase 29

Este paper es el **andamiaje conceptual** de la [Clase 29](/clases/clase-29) (Modelos Generativos en Visión), no una referencia secundaria:

- **La slide del trilema.** La clase incluye una slide "Generative learning trilemma" que enlaza directamente a este trabajo; el triángulo con los vértices *calidad / velocidad / cobertura* es literalmente la Fig. 1 de Xiao et al.
- **La tabla VAE / GAN / Difusión es el trilema.** La comparación que la clase hace de las tres familias sobre velocidad / calidad / distribución operacionaliza exactamente el diagnóstico del paper. Sitúa todo el panorama de [modelos generativos](/fundamentos/modelos-generativos) en una sola lente.
- **Bisagra entre GANs y difusión.** La clase recorre las [GANs](/papers/gan-goodfellow-2014) y los [modelos de difusión](/papers/ddpm-ho-2020); este paper es el que **une ambos mundos**, usando el andamiaje de DDPM pero reemplazando la gaussiana del denoising por un generador adversarial. Es el ejemplo canónico de que las familias generativas no son compartimentos estancos.
- **Por qué la difusión es lenta, en su raíz.** Para el fundamento de [modelos de difusión](/fundamentos/modelos-de-difusion), aporta la explicación *fundamental* (no solo empírica) de la lentitud: el supuesto gaussiano del denoising solo vale con pasos infinitesimales. Esa es la pieza teórica que motiva todas las técnicas de muestreo acelerado.

## Notas y enlaces

- Preprint: [arXiv:2112.07804](https://arxiv.org/abs/2112.07804) (v2, abril 2022).
- Sitio y código: [nvlabs.github.io/denoising-diffusion-gan](https://nvlabs.github.io/denoising-diffusion-gan).
- Venue: ICLR 2022 (conference paper).
- Afiliaciones: The University of Chicago (trabajo durante pasantía en NVIDIA), NVIDIA.
