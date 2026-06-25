# Classifier-Free Diffusion Guidance — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Classifier-Free Diffusion Guidance*.
- **Autores:** Jonathan Ho y Tim Salimans (Google Research, Brain team).
- **Venue:** Una versión corta apareció en el *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications* ([openreview.net/pdf?id=qw8AKxfYbI](https://openreview.net/pdf?id=qw8AKxfYbI)).
- **Preprint:** arXiv:2207.12598v1, 26 jul 2022, [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598).
- **Código / modelos:** no se libera código oficial con el paper; el mecanismo es tan simple (un cambio de una línea en entrenamiento y otro en sampling) que fue reimplementado por toda la comunidad casi de inmediato.

Este paper resuelve un problema concreto que había dejado abierto *Diffusion Models Beat GANs* de Dhariwal & Nichol (2021): cómo conseguir el trade-off entre **fidelidad** (calidad de cada muestra) y **diversidad** (cobertura de modos) en modelos de difusión —el equivalente al *truncation trick* de BigGAN o al *low temperature sampling* de Glow— **sin necesitar un clasificador externo**. La técnica de Dhariwal & Nichol, *classifier guidance*, lograba ese trade-off pero a costa de entrenar un clasificador de imágenes adicional, y —crucialmente— ese clasificador debía entrenarse sobre **imágenes ruidosas** $z_\lambda$ (las versiones corrompidas que ve el modelo de difusión en cada paso), de modo que no se podía enchufar un clasificador preentrenado estándar.

La tesis de Ho & Salimans: la guía se puede realizar con un **modelo puramente generativo**, sin clasificador alguno. La receta —que ellos bautizan *classifier-free guidance*— es entrenar conjuntamente un modelo de difusión **condicional** $\epsilon_\theta(z_\lambda, c)$ y uno **no-condicional** $\epsilon_\theta(z_\lambda)$ usando una sola red (basta con pasar un token nulo $\varnothing$ en lugar de la etiqueta de clase con cierta probabilidad), y en sampling combinar linealmente ambas estimaciones de score para extrapolar en la dirección que se aleja de lo no-condicional y se acerca a lo condicional. Barriendo el peso de esa combinación —la *guidance scale*— se obtiene una curva FID/IS equivalente a la de classifier guidance.

Para la Clase 29 (Modelos Generativos en Visión) esto importa porque **es el mecanismo estándar de todos los modelos text-to-image modernos** —Stable Diffusion, DALL-E 2, Imagen— y porque el `guidance_scale` que el laboratorio ajusta al llamar a la pipeline de `diffusers` es, literalmente, el parámetro $w$ de este paper. Entender este paper es entender *por qué* subir la guidance hace que la imagen "obedezca" más al prompt y *por qué*, si se sube demasiado, las imágenes se saturan y pierden variedad.

## 2. Contexto histórico: del *truncation trick* a classifier guidance

Hacia 2020–2021 los modelos de difusión (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020 —el paper de DDPM— Song et al., 2021) habían emergido como una familia generativa expresiva y flexible, con calidad competitiva en síntesis de imagen y audio. En ImageNet ya superaban a BigGAN-deep (Brock et al., 2019) y VQ-VAE-2 (Razavi et al., 2019) en FID y en *classification accuracy score* (Ho et al., 2021; Dhariwal & Nichol, 2021).

Pero les faltaba una palanca que las GANs y los flujos sí tenían: el muestreo "a baja temperatura". En BigGAN, el *truncation trick* recorta el rango del ruido de entrada y produce una curva de trade-off entre FID e Inception Score (IS) —más truncamiento = más fidelidad, menos diversidad. En Glow (Kingma & Dhariwal, 2018), bajar la temperatura del muestreo tiene el mismo efecto. El problema, documentado por Dhariwal & Nichol, es que los intentos ingenuos de replicar esto en difusión —escalar los vectores de score del modelo o reducir la varianza del ruido gaussiano del proceso inverso— **no funcionan**: producen muestras borrosas y de baja calidad.

**Classifier guidance** (Dhariwal & Nichol, 2021) fue la solución. La idea: mezclar la estimación de score del modelo de difusión con el gradiente del log-likelihood de un clasificador auxiliar $p_\theta(c \mid z_\lambda)$:

$$\tilde\epsilon_\theta(z_\lambda, c) = \epsilon_\theta(z_\lambda, c) - w\,\sigma_\lambda\,\nabla_{z_\lambda}\log p_\theta(c \mid z_\lambda) \approx -\sigma_\lambda\,\nabla_{z_\lambda}\big[\log p(z_\lambda \mid c) + w\log p_\theta(c \mid z_\lambda)\big]$$

El efecto es muestrear de una distribución modificada $\tilde p_\theta(z_\lambda \mid c) \propto p_\theta(z_\lambda \mid c)\,p_\theta(c \mid z_\lambda)^w$: se sobreponderan los datos a los que el clasificador asigna alta probabilidad de la etiqueta correcta. Como los datos "bien clasificables" puntúan alto en IS por diseño, subir $w > 0$ mejora el IS a costa de la diversidad. El parámetro $w$ es la fuerza de la guía.

El paper enumera **tres objeciones** a classifier guidance que motivan buscar una alternativa:

1. **Complica el pipeline.** Hay que entrenar un clasificador extra, y debe entrenarse sobre datos ruidosos $z_\lambda$, así que no se puede usar uno preentrenado estándar.
2. **Parece un ataque adversarial.** Como el sampling da pasos en la dirección del gradiente del clasificador, el procedimiento se puede interpretar como un *ataque adversarial basado en gradiente* contra ese clasificador. Esto abre la duda incómoda de si la mejora en FID/IS es real o si simplemente el modelo está aprendiendo a "engañar" a las métricas (que también se calculan con clasificadores tipo Inception).
3. **Parece GAN.** Dar pasos en la dirección de gradientes de clasificador recuerda al entrenamiento de una GAN, otra familia que ya puntúa bien en métricas basadas en clasificadores —lo que refuerza la sospecha anterior.

Resolver estas dudas exige una guía que **no use ningún clasificador**. Esa es la contribución.

## 3. Contribución central

La aportación es **classifier-free guidance**: un método para obtener el mismo trade-off fidelidad/diversidad que classifier guidance pero **sin entrenar ni invocar clasificador alguno**. En vez de entrenar un clasificador separado, se entrena un modelo de difusión **no-condicional** $p_\theta(z)$ (parametrizado por el score $\epsilon_\theta(z_\lambda)$) **junto con** el modelo condicional $p_\theta(z \mid c)$ (parametrizado por $\epsilon_\theta(z_\lambda, c)$), usando **una sola red**. El modelo no-condicional se obtiene simplemente pasando un **token de clase nulo** $\varnothing$: $\epsilon_\theta(z_\lambda) = \epsilon_\theta(z_\lambda, c = \varnothing)$.

El entrenamiento conjunto se logra fijando $c \leftarrow \varnothing$ con probabilidad $p_\text{uncond}$ (un hiperparámetro), es decir, haciendo **dropout del condicionamiento**. Los autores eligen entrenamiento conjunto —en vez de dos modelos separados— porque "es extremadamente simple de implementar, no complica el pipeline de entrenamiento y no aumenta el número total de parámetros".

En sampling se forma el score guiado como combinación lineal:

$$\tilde\epsilon_\theta(z_\lambda, c) = (1 + w)\,\epsilon_\theta(z_\lambda, c) - w\,\epsilon_\theta(z_\lambda)$$

Esta es la ecuación (6) del paper, el corazón de todo. (En la notación más difundida de Stable Diffusion se reescribe como $\tilde\epsilon = \epsilon_\text{uncond} + s\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$ con *guidance scale* $s = w + 1$; es la misma fórmula, solo cambia la convención del peso. Con $w = 0$, o $s = 1$, no hay guía y se obtiene el modelo condicional puro.) El término $-w\,\epsilon_\theta(z_\lambda)$ **resta la dirección no-condicional** y extrapola más allá del condicional: empuja la muestra hacia donde el condicionamiento $c$ "importa" y la aleja de lo que el modelo generaría sin condicionar.

Lo notable es que la ecuación (6) **no contiene ningún gradiente de clasificador**, así que el paso en la dirección $\tilde\epsilon_\theta$ no puede interpretarse como un ataque adversarial —respondiendo a la objeción 2 del §2. Esto, según los autores, demuestra que un modelo puramente generativo puede maximizar las métricas basadas en clasificador (FID, IS) **sin** recurrir a gradientes de clasificador.

## 4. Método

### 4.1. Marco de difusión en tiempo continuo

El paper trabaja en tiempo continuo (Song et al., 2021; Kingma et al., 2021). El proceso forward es *variance-preserving*: $q(z_\lambda \mid x) = \mathcal{N}(\alpha_\lambda x, \sigma_\lambda^2 I)$, con $\alpha_\lambda^2 = 1/(1 + e^{-\lambda})$ y $\sigma_\lambda^2 = 1 - \alpha_\lambda^2$. La variable $\lambda = \log(\alpha_\lambda^2/\sigma_\lambda^2)$ se interpreta como el **log signal-to-noise ratio** (log-SNR) de $z_\lambda$; el forward corre en dirección de $\lambda$ decreciente (más ruido). El modelo se parametriza con **$\epsilon$-prediction** (como en DDPM): $x_\theta(z_\lambda) = (z_\lambda - \sigma_\lambda\,\epsilon_\theta(z_\lambda))/\alpha_\lambda$, y se entrena con el objetivo de *denoising score matching*

$$\mathbb{E}_{\epsilon,\lambda}\big[\lVert \epsilon_\theta(z_\lambda) - \epsilon \rVert_2^2\big],$$

donde $\epsilon \sim \mathcal{N}(0, I)$ y $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$. La conexión clave es que $\epsilon_\theta(z_\lambda) \approx -\sigma_\lambda\,\nabla_{z_\lambda}\log p(z_\lambda)$: la red de denoising **es** un estimador del gradiente del log-density de los datos ruidosos (el *score*). Para condicionar, la única modificación es que la red recibe $c$ como entrada adicional: $\epsilon_\theta(z_\lambda, c)$.

### 4.2. Entrenamiento conjunto (Algoritmo 1)

El entrenamiento es idéntico al de un modelo de difusión condicional normal salvo por **una línea**: tras muestrear el par $(x, c)$ del dataset, se descarta el condicionamiento poniendo $c \leftarrow \varnothing$ con probabilidad $p_\text{uncond}$. Luego se muestrea $\lambda \sim p(\lambda)$, se corrompe $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$ y se da un paso de gradiente sobre $\lVert \epsilon_\theta(z_\lambda, c) - \epsilon \rVert^2$. Esa única red aprende, con el mismo conjunto de pesos, tanto a denoiser condicionado por la clase como a denoiser "a ciegas" cuando ve el token nulo.

### 4.3. Sampling guiado (Algoritmo 2)

En cada paso $t$ de la secuencia creciente de log-SNR $\lambda_1 < \cdots < \lambda_T$, se forma el score guiado con la ecuación (6), $\tilde\epsilon_t = (1+w)\epsilon_\theta(z_t, c) - w\,\epsilon_\theta(z_t)$, se reconstruye $\tilde x_t = (z_t - \sigma_{\lambda_t}\tilde\epsilon_t)/\alpha_{\lambda_t}$ y se da el paso del sampler ancestral (que, señalan los autores, podría reemplazarse por otro sampler como DDIM). El costo: **cada paso de sampling requiere evaluar la red dos veces**, una para el score condicional y otra para el no-condicional.

### 4.4. El clasificador implícito

Aunque la ecuación (6) no usa ningún clasificador, está *inspirada* en el gradiente de un **clasificador implícito** $p^i(c \mid z_\lambda) \propto p(z_\lambda \mid c)/p(z_\lambda)$ (regla de Bayes aplicada al modelo generativo). Si se tuvieran los scores exactos, el gradiente de ese clasificador implícito sería $\nabla_{z_\lambda}\log p^i(c \mid z_\lambda) = -\frac{1}{\sigma_\lambda}[\epsilon^*(z_\lambda, c) - \epsilon^*(z_\lambda)]$, y aplicarle classifier guidance recuperaría una expresión análoga a la (6). Pero los autores son cuidadosos: $\tilde\epsilon_\theta$ se construye con scores **estimados por redes neuronales sin restricciones**, que en general **no** forman campos vectoriales conservativos, así que **no existe ningún clasificador real cuyo gradiente sea la ecuación (6)**. Citan a Grandvalet & Bengio (2004) y Grünwald & Langford (2007) para advertir que invertir un modelo generativo vía Bayes no garantiza un buen clasificador —especialmente con modelos mal especificados. Por eso la justificación final es **empírica**, no teórica: funciona en la práctica (§5).

## 5. Experimentos

Los autores entrenan modelos de difusión con classifier-free guidance sobre **ImageNet class-conditional** a resoluciones $64\times64$ y $128\times128$ —el banco de pruebas estándar para estudiar el trade-off FID/IS desde BigGAN. Usan las mismas arquitecturas e hiperparámetros que los modelos guiados de Dhariwal & Nichol (salvo el entrenamiento en tiempo continuo); de hecho, al amortizar el modelo condicional y el no-condicional en una sola red sin clasificador extra, usan **menos capacidad** que el trabajo previo. El propósito es servir de prueba de concepto, no de empujar el estado del arte.

### 5.1. Barrido de la fuerza de guía $w$

Barriendo $w \in \{0, 0.1, 0.2, \dots, 4\}$ y calculando FID e IS con 50.000 muestras, observan el resultado central: **un trade-off claro y monótono** —al subir $w$, el **FID empeora** (sube) y el **IS mejora** (sube). El mejor FID se obtiene con poca guía ($w = 0.1$ o $0.3$ según el dataset) y el mejor IS con guía fuerte ($w \geq 4$). En ImageNet $64\times64$ con $p_\text{uncond}=0.1$: $w=0$ da FID 1.8 / IS 53.7; $w=0.1$ da FID 1.55 / IS 66.1 (mejor FID); $w=4$ da FID 26.2 / IS 260.2 (mejor IS). En $128\times128$, con $w=0.3$ el FID supera al ADM-G guiado por clasificador, y con $w=4$ supera a BigGAN-deep en FID *e* IS simultáneamente —resultados que los autores describen como estado del arte en ese momento. Visualmente, subir la guía reduce la variedad de las muestras y aumenta la fidelidad individual, **confirmando empíricamente** que classifier-free guidance replica el efecto de classifier guidance / truncation. (Las muestras fuertemente guiadas, observan, muestran colores saturados —un artefacto característico.)

### 5.2. Probabilidad de entrenamiento no-condicional $p_\text{uncond}$

El hiperparámetro principal de entrenamiento es $p_\text{uncond}$. Entrenan con $p_\text{uncond} \in \{0.1, 0.2, 0.5\}$ y hallan que **$0.5$ rinde consistentemente peor** que $0.1$ y $0.2$ en toda la frontera IS/FID, mientras que $0.1$ y $0.2$ rinden aproximadamente igual. Conclusión: **basta dedicar una porción pequeña de la capacidad del modelo** a la tarea no-condicional para producir scores guiados efectivos. Curiosamente, esto refleja un hallazgo paralelo de Dhariwal & Nichol con classifier guidance: clasificadores relativamente pequeños bastan para una guía efectiva.

### 5.3. Número de pasos de sampling $T$

Variando $T \in \{128, 256, 1024\}$ en el modelo $128\times128$, la calidad mejora al subir $T$, y $T=256$ logra un buen balance calidad/velocidad. Importante para comparaciones justas: como cada paso requiere **dos** evaluaciones de la red (condicional + no-condicional), el setting comparable en velocidad a ADM-G (que usa ~256 pasos con una sola evaluación) sería $T=128$, donde el modelo queda por debajo de ADM-G en FID.

## 6. Discusión e interpretación

La ventaja más práctica que destacan los autores es la **extrema simplicidad**: classifier-free guidance es un cambio de una sola línea en entrenamiento (dropout del condicionamiento) y otro en sampling (mezclar los dos scores), sin clasificador extra ni complicación del pipeline.

Aportan además una **explicación intuitiva** de cómo funciona la guía: *decrementa la verosimilitud no-condicional de la muestra mientras incrementa la condicional*. Classifier-free guidance lo consigue con un **término de score negativo** (el $-w\,\epsilon_\theta(z_\lambda)$), un mecanismo que —señalan— no se había explorado antes y podría tener usos en otras aplicaciones. Intuitivamente: el modelo es empujado hacia regiones donde "la clase $c$ explica los datos mucho mejor que la ausencia de clase".

Mencionan también una variante: si la distribución de clases es conocida y hay pocas clases, se puede obtener el score no-condicional a partir de los condicionales vía $\sum_c p(x \mid c)p(c) = p(x)$, evitando entrenar el modelo no-condicional —aunque esto requiere tantas pasadas como clases haya, inviable para condicionamiento de alta dimensión (como texto).

## 7. Limitaciones reconocidas

- **Costo de sampling (doble forward pass).** Classifier-free guidance necesita **dos evaluaciones** del modelo de difusión por paso (condicional + no-condicional). Como los clasificadores suelen ser más pequeños que los generadores, classifier guidance puede ser *más rápido* en sampling. Los autores sugieren mitigarlo inyectando el condicionamiento tarde en la red, pero lo dejan como trabajo futuro.
- **Pérdida de diversidad con $w$ alto.** Por construcción, subir la guía aumenta la fidelidad a costa de la diversidad. Los autores plantean la pregunta ética/práctica de si esa pérdida de diversidad es aceptable: en despliegues reales puede haber **impactos negativos** cuando ciertas partes de los datos están subrepresentadas, porque la guía fuerte las suprime aún más. Mejorar la calidad *manteniendo* la diversidad lo señalan como dirección abierta.
- **Sin garantías teóricas.** Como el clasificador implícito (§4.4) no existe en sentido estricto (los scores no son campos conservativos), no hay garantías formales sobre la calidad de la señal de guía; la justificación es puramente empírica.

## 8. Impacto

Pese a su modestia ("prueba de concepto"), classifier-free guidance se convirtió en **el mecanismo de guía estándar de toda la generación condicional moderna por difusión**. Es el componente que hace que los modelos text-to-image obedezcan al prompt:

- **Stable Diffusion** (Rombach et al., 2022) lo usa en el espacio latente; el `guidance_scale` de la pipeline de `diffusers` es exactamente el $s = w + 1$ de este paper.
- **DALL-E 2** (Ramesh et al., 2022) e **Imagen** (Saharia et al., 2022) lo adoptan; Imagen en particular reporta que la guía fuerte sobre texto es esencial para la alineación prompt-imagen, e introduce *dynamic thresholding* precisamente para contrarrestar la saturación de color que este paper ya había observado en §5.1.

En la práctica diaria, la "guidance scale" que cualquier usuario ajusta en una interfaz de generación de imágenes —típicamente entre 7 y 12 para Stable Diffusion— **es este parámetro**. Subirlo fuerza la adherencia al prompt (más fidelidad, menos variedad); bajarlo da resultados más diversos pero menos fieles. La negociación calidad/diversidad que Ho & Salimans formalizaron es, hoy, la perilla más usada de la generación de imágenes.

## 9. Conexión con la Clase 29 (Modelos Generativos en Visión)

El laboratorio de la Clase 29 usa **Stable Diffusion** a través de la librería `diffusers`, y el parámetro `guidance_scale` que se pasa a la pipeline (p. ej. `pipe(prompt, guidance_scale=7.5)`) **es justo el peso de este paper**. Entender Ho & Salimans (2022) es entender qué ocurre por dentro cuando se mueve esa perilla:

- **Por qué el prompt "fuerza" la imagen.** Sin guía ($s = 1$, $w = 0$), Stable Diffusion muestrea del modelo condicional puro y suele ignorar parcialmente el prompt. La fórmula $\tilde\epsilon = \epsilon_\text{uncond} + s\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$ **extrapola** en la dirección $(\epsilon_\text{cond} - \epsilon_\text{uncond})$ —que es, precisamente, "lo que el texto aporta" sobre la generación a ciegas— amplificándolo por el factor $s$. Por eso subir `guidance_scale` hace que la imagen se parezca cada vez más a lo que pide el prompt: se está exagerando el componente condicional del score en cada paso de denoising.
- **El doble forward pass que ve el estudiante.** Cada paso de sampling en `diffusers` ejecuta la U-Net **dos veces** —una con el embedding del prompt y otra con un embedding vacío (el token nulo $\varnothing$ de este paper, implementado como el *unconditional / negative prompt*)—. Eso explica por qué el sampling con guía cuesta el doble que sin guía, y conecta directamente con la limitación del §7.
- **El trade-off que el lab puede medir.** Si en el lab se barre `guidance_scale`, se reproduce a escala el experimento del §5.1: valores bajos dan imágenes diversas pero a veces poco fieles al prompt; valores altos dan imágenes muy fieles pero saturadas, repetitivas y con artefactos —el mismo colapso de diversidad y la misma saturación de color que Ho & Salimans documentaron en ImageNet.
- **El *negative prompt* como caso del término no-condicional.** El truco del *negative prompt* de Stable Diffusion es una generalización directa de la ecuación (6): en lugar de restar el score del token nulo, se resta el score condicionado a un prompt "negativo", empujando la imagen *lejos* de ese concepto. Es la misma maquinaria de "restar una dirección de score" que el paper introdujo.

Referencias internas del curso: el marco de $\epsilon$-prediction y *denoising score matching* sobre el que se construye este método viene de [DDPM (Ho et al., 2020)](/papers/ho-ddpm-2020); la aplicación en espacio latente que populariza la guía es [Latent Diffusion / Stable Diffusion (Rombach et al., 2022)](/papers/rombach-latentdiffusion-2022); el fundamento transversal está en [Modelos de difusión](/fundamentos/modelos-de-difusion); y el hub de la sesión es [Clase 29](/clases/clase-29).
