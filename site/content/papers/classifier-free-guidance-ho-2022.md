---
title: "Classifier-Free Diffusion Guidance (2022)"
weight: 338
math: true
---

{{< paper-card
    title="Classifier-Free Diffusion Guidance"
    authors="Jonathan Ho, Tim Salimans"
    year="2022"
    venue="NeurIPS 2021 Workshop / arXiv"
    pdf="/papers/classifier-free-guidance-ho-2022.pdf"
    arxiv="2207.12598" >}}
Paper de Google Research que introduce **classifier-free guidance**, el mecanismo de guía estándar de toda la generación condicional moderna por difusión. La idea es elegante: entrenar una **sola red** que sea a la vez modelo de difusión **condicional** y **no-condicional** (haciendo *dropout* del condicionamiento durante el entrenamiento), y en *sampling* combinar linealmente ambas estimaciones de *score* para extrapolar hacia lo condicional. Con esto se obtiene el mismo trade-off **fidelidad/diversidad** que la *classifier guidance* de Dhariwal & Nichol, pero **sin entrenar ni invocar un clasificador externo**. El peso de esa combinación —la *guidance scale* $w$— es exactamente el `guidance_scale` que el laboratorio de la [Clase 29](/clases/clase-29) ajusta al llamar a Stable Diffusion.
{{< /paper-card >}}

---

## Contexto: del *truncation trick* a la *classifier guidance*

Hacia 2020-2021 los modelos de difusión (Sohl-Dickstein 2015; Song & Ermon 2019; [DDPM, Ho et al. 2020](/papers/ddpm-ho-2020); Song et al. 2021) ya superaban a BigGAN-deep y VQ-VAE-2 en FID sobre ImageNet. Pero les faltaba una palanca que las GANs y los flujos sí tenían: el muestreo "a baja temperatura". En BigGAN, el *truncation trick* recorta el rango del ruido de entrada y produce una curva de trade-off entre FID e Inception Score (IS) —más truncamiento = más fidelidad, menos diversidad—; en Glow, bajar la temperatura del *sampling* logra lo mismo. El problema, documentado por Dhariwal & Nichol, es que los intentos ingenuos de replicarlo en difusión (escalar los *scores* o reducir la varianza del ruido) **no funcionan**: producen muestras borrosas.

**Classifier guidance** (Dhariwal & Nichol, 2021) fue la solución: mezclar el *score* del modelo de difusión con el gradiente del log-likelihood de un clasificador auxiliar $p_\theta(c \mid z_\lambda)$, muestreando así de una distribución modificada $\tilde p(z_\lambda \mid c) \propto p(z_\lambda \mid c)\,p_\theta(c \mid z_\lambda)^w$. Funcionaba, pero con tres objeciones que motivan buscar una alternativa:

1. **Complica el pipeline.** Hay que entrenar un clasificador extra que, crucialmente, debe entrenarse sobre **imágenes ruidosas** $z_\lambda$ (las versiones corrompidas que ve la difusión), así que no sirve uno preentrenado estándar.
2. **Parece un ataque adversarial.** El *sampling* da pasos en la dirección del gradiente del clasificador, lo que se puede interpretar como un ataque adversarial basado en gradiente —y abre la duda de si la mejora en FID/IS es real o si el modelo aprende a "engañar" a las métricas (que también usan clasificadores tipo Inception).
3. **Parece GAN.** Dar pasos en dirección de gradientes de clasificador recuerda al entrenamiento de una GAN, reforzando la sospecha anterior.

Resolver estas dudas exige una guía que **no use ningún clasificador**.

## Contribución central: entrenar un solo modelo

En lugar de entrenar un clasificador separado, se entrena un modelo de difusión **no-condicional** $\epsilon_\theta(z_\lambda)$ **junto con** el condicional $\epsilon_\theta(z_\lambda, c)$ usando **una sola red**. El modelo no-condicional se obtiene simplemente pasando un **token de clase nulo** $\varnothing$:

$$\epsilon_\theta(z_\lambda) = \epsilon_\theta(z_\lambda, c = \varnothing)$$

El entrenamiento conjunto se logra fijando $c \leftarrow \varnothing$ con probabilidad $p_\text{uncond}$ (un hiperparámetro), es decir, haciendo **dropout del condicionamiento**. Los autores eligen entrenamiento conjunto —en vez de dos modelos separados— porque "es extremadamente simple de implementar, no complica el pipeline de entrenamiento y no aumenta el número total de parámetros".

En *sampling* se forma el *score* guiado como combinación lineal (la ecuación 6 del paper, el corazón de todo):

$$\tilde\epsilon_\theta(z_\lambda, c) = (1 + w)\,\epsilon_\theta(z_\lambda, c) - w\,\epsilon_\theta(z_\lambda)$$

En la notación más difundida de Stable Diffusion se reescribe como $\tilde\epsilon = \epsilon_\text{uncond} + s\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$ con *guidance scale* $s = w + 1$; es la misma fórmula, solo cambia la convención del peso. Con $w = 0$ (o $s = 1$) no hay guía y se obtiene el modelo condicional puro. El término $-w\,\epsilon_\theta(z_\lambda)$ **resta la dirección no-condicional** y extrapola más allá del condicional: empuja la muestra hacia donde el condicionamiento $c$ "importa" y la aleja de lo que el modelo generaría a ciegas.

Lo notable es que la ecuación (6) **no contiene ningún gradiente de clasificador**, así que el paso no puede interpretarse como un ataque adversarial —respondiendo a la objeción 2. Esto demuestra que un modelo puramente generativo puede maximizar FID e IS **sin** recurrir a gradientes de clasificador.

## Método

El marco es difusión en tiempo continuo con proceso forward *variance-preserving* $q(z_\lambda \mid x) = \mathcal{N}(\alpha_\lambda x, \sigma_\lambda^2 I)$, donde $\lambda = \log(\alpha_\lambda^2/\sigma_\lambda^2)$ es el **log signal-to-noise ratio**. El modelo se parametriza con **$\epsilon$-prediction** (como en [DDPM](/papers/ddpm-ho-2020)) y se entrena con *denoising score matching* $\mathbb{E}_{\epsilon,\lambda}[\lVert \epsilon_\theta(z_\lambda) - \epsilon \rVert_2^2]$. La conexión clave es que $\epsilon_\theta(z_\lambda) \approx -\sigma_\lambda\,\nabla_{z_\lambda}\log p(z_\lambda)$: la red de *denoising* **es** un estimador del *score*.

- **Entrenamiento (Algoritmo 1).** Idéntico al de un modelo de difusión condicional normal salvo por **una línea**: tras muestrear el par $(x, c)$, se descarta el condicionamiento poniendo $c \leftarrow \varnothing$ con probabilidad $p_\text{uncond}$. Esa única red aprende, con el mismo conjunto de pesos, tanto a hacer *denoising* condicionado como "a ciegas" al ver el token nulo.
- **Sampling guiado (Algoritmo 2).** En cada paso se forma $\tilde\epsilon_t$ con la ecuación (6), se reconstruye $\tilde x_t$ y se da el paso del *sampler* ancestral (intercambiable por DDIM u otro). El costo: **cada paso requiere evaluar la red dos veces**, una para el *score* condicional y otra para el no-condicional.

### El clasificador implícito

La ecuación (6) está *inspirada* en el gradiente de un **clasificador implícito** $p^i(c \mid z_\lambda) \propto p(z_\lambda \mid c)/p(z_\lambda)$ (regla de Bayes aplicada al modelo generativo). Pero los autores son cuidadosos: $\tilde\epsilon_\theta$ se construye con *scores* estimados por redes sin restricciones, que en general **no** forman campos vectoriales conservativos, así que **no existe ningún clasificador real cuyo gradiente sea la ecuación (6)**. La justificación final es **empírica**, no teórica: funciona en la práctica.

## Experimentos

Los autores entrenan sobre **ImageNet class-conditional** a $64\times64$ y $128\times128$, el banco de pruebas estándar para el trade-off FID/IS desde BigGAN, con menos capacidad que el trabajo previo (al amortizar condicional y no-condicional en una sola red sin clasificador extra).

- **Barrido de la fuerza de guía $w$.** Barriendo $w \in \{0, 0.1, \dots, 4\}$, el resultado central es un **trade-off claro y monótono**: al subir $w$, el **FID empeora** y el **IS mejora**. En ImageNet $64\times64$ con $p_\text{uncond}=0.1$: $w=0$ da FID 1.8 / IS 53.7; $w=0.1$ da el mejor FID (1.55 / IS 66.1); $w=4$ da el mejor IS (FID 26.2 / IS 260.2). En $128\times128$, $w=0.3$ supera al ADM-G guiado por clasificador, y $w=4$ supera a BigGAN-deep en FID *e* IS simultáneamente. Las muestras fuertemente guiadas muestran **colores saturados** —un artefacto característico.
- **Probabilidad no-condicional $p_\text{uncond}$.** Con $p_\text{uncond} \in \{0.1, 0.2, 0.5\}$, el valor $0.5$ rinde consistentemente peor; $0.1$ y $0.2$ rinden igual de bien. **Basta dedicar una porción pequeña de la capacidad** a la tarea no-condicional.
- **Pasos de sampling $T$.** $T=256$ logra un buen balance. Como cada paso requiere **dos** evaluaciones de la red, el setting comparable en velocidad a ADM-G (256 pasos, una evaluación) sería $T=128$, donde el modelo queda por debajo.

## Limitaciones

- **Costo de sampling (doble forward pass).** Necesita dos evaluaciones del modelo por paso; como los clasificadores suelen ser más pequeños, la *classifier guidance* puede ser *más rápida* en *sampling*.
- **Pérdida de diversidad con $w$ alto.** Subir la guía aumenta la fidelidad a costa de la diversidad; los autores advierten del impacto negativo cuando partes subrepresentadas de los datos quedan aún más suprimidas.
- **Sin garantías teóricas.** Como el clasificador implícito no existe en sentido estricto, no hay garantías formales sobre la calidad de la señal de guía.

## Impacto

Pese a presentarse como "prueba de concepto", classifier-free guidance se convirtió en **el mecanismo de guía estándar de toda la generación condicional moderna por difusión** —el componente que hace que los modelos text-to-image obedezcan al prompt:

- **[Stable Diffusion](/papers/latent-diffusion-rombach-2022)** (Rombach et al., 2022) lo usa en el espacio latente; el `guidance_scale` de la pipeline de `diffusers` es exactamente el $s = w + 1$ de este paper.
- **DALL-E 2** (Ramesh et al., 2022) e **Imagen** (Saharia et al., 2022) lo adoptan; Imagen reporta que la guía fuerte sobre texto es esencial para la alineación prompt-imagen e introduce *dynamic thresholding* para contrarrestar la saturación de color que este paper ya había observado.

En la práctica diaria, la "guidance scale" que cualquier usuario ajusta en una interfaz de generación —típicamente entre 7 y 12 para Stable Diffusion— **es este parámetro**. Subirlo fuerza la adherencia al prompt (más fidelidad, menos variedad); bajarlo da resultados más diversos pero menos fieles.

## Por qué importa para la Clase 29

El laboratorio de la [Clase 29](/clases/clase-29) usa **Stable Diffusion** a través de `diffusers`, y el `guidance_scale` que se pasa a la pipeline (p. ej. `pipe(prompt, guidance_scale=7.5)`) **es justo el peso $w$ de este paper**:

- **Por qué el prompt "fuerza" la imagen.** La fórmula $\tilde\epsilon = \epsilon_\text{uncond} + s\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$ **extrapola** en la dirección $(\epsilon_\text{cond} - \epsilon_\text{uncond})$ —"lo que el texto aporta" sobre la generación a ciegas— amplificándolo por $s$. Por eso subir `guidance_scale` hace que la imagen se parezca cada vez más al prompt.
- **El doble forward pass.** Cada paso de *sampling* ejecuta la U-Net **dos veces** —una con el embedding del prompt y otra con un embedding vacío (el token nulo $\varnothing$, implementado como el *negative prompt*)—, lo que explica por qué el *sampling* con guía cuesta el doble.
- **El trade-off que el lab puede medir.** Barrer `guidance_scale` reproduce a escala el experimento del paper: valores bajos dan imágenes diversas pero poco fieles; valores altos dan imágenes muy fieles pero saturadas y repetitivas.
- **El *negative prompt*.** Es una generalización directa de la ecuación (6): en lugar de restar el *score* del token nulo, se resta el *score* condicionado a un prompt "negativo", empujando la imagen *lejos* de ese concepto.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/2207.12598 (v1, 26 jul 2022)
- Versión corta: NeurIPS 2021 Workshop on Deep Generative Models — https://openreview.net/pdf?id=qw8AKxfYbI
- Marco de $\epsilon$-prediction y *denoising score matching*: [DDPM (Ho et al., 2020)](/papers/ddpm-ho-2020).
- Aplicación en espacio latente que populariza la guía: [Latent Diffusion / Stable Diffusion (Rombach et al., 2022)](/papers/latent-diffusion-rombach-2022).
- Fundamento transversal: [Modelos de difusión](/fundamentos/modelos-de-difusion).
- Hub de la sesión: [Clase 29](/clases/clase-29).
