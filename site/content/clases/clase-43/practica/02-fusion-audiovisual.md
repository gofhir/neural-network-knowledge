---
title: "02 - Fusión audiovisual bajo ruido"
weight: 20
math: true
---

> La clase pregunta *"¿por qué necesitamos video si tenemos el audio?"* y responde con una figura: la curva de exactitud contra relación señal-ruido, donde **la línea del video es horizontal**. Este camino reconstruye esa figura desde cero, mide de dónde sale la horizontalidad, por qué el canal visual tiene un techo que no baja con más datos, y en qué condiciones la fusión **empeora** el resultado.

---

## 1. El montaje

Para aislar el fenómeno hay que controlar exactamente qué degrada a qué. El montaje: 40 clases, dos modalidades, y una asimetría deliberada.

```python
K, N, D = 40, 20000, 24
proto_a = rng.normal(0, 1, (K, D))     # prototipo acústico por clase
proto_v = rng.normal(0, 1, (K, D))     # prototipo visual por clase

# El canal VISUAL es intrínsecamente ambiguo: los visemas colapsan fonemas
# distintos ('p', 'b', 'm' se ven igual). Pares de clases comparten prototipo.
for k in range(0, K, 2):
    proto_v[k + 1] = proto_v[k] + rng.normal(0, 0.35, D)

sigma_v = 1.0                          # ruido visual FIJO
def genera(snr_db):
    sigma_a = 3.2 * 10 ** (-snr_db / 20)   # solo el ruido ACÚSTICO varía
    ...
```

Dos decisiones que reproducen la física del problema real:

- **El ruido visual no depende del SNR acústico.** Un ruido de fondo no borronea la imagen. Esa independencia es toda la razón de ser de la fusión.
- **El canal visual tiene ambigüedad estructural.** Los pares de clases con prototipo casi idéntico son el análogo de los visemas: `/p/`, `/b/` y `/m/` producen la misma configuración de labios porque lo que los separa —sonoridad, nasalidad— ocurre dentro de la garganta.

La clasificación es la log-verosimilitud bajo ruido gaussiano isotrópico, que da los logits directamente:

```python
def clasifica_gauss(X, proto, sigma):
    d2 = ((X[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    return -d2 / (2 * sigma ** 2)
```

## 2. Las dos fusiones, en cuatro backends

```python
fusion_optima = la + lv                                     # suma de log-verosimilitudes
fusion_tardia = np.log(0.5*softmax(la) + 0.5*softmax(lv))   # promedio de probabilidades
```

{{< tabs >}}
{{< tab name="NumPy" >}}
```python
def fus_suma(la, lv):
    return la + lv

def fus_promedio(la, lv):
    return np.log(0.5 * softmax(la) + 0.5 * softmax(lv))
```
{{< /tab >}}
{{< tab name="PyTorch" >}}
```python
suma = la + lv
prom = torch.log(0.5 * torch.softmax(la, -1) + 0.5 * torch.softmax(lv, -1))
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
suma = la + lv
prom = tf.math.log(0.5 * tf.nn.softmax(la) + 0.5 * tf.nn.softmax(lv))
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
suma = la + lv
prom = jnp.log(0.5 * jax.nn.softmax(la) + 0.5 * jax.nn.softmax(lv))
```
{{< /tab >}}
{{< /tabs >}}

```
  PyTorch      suma max|dif| = 0.00e+00   promedio max|dif| = 8.88e-16
  TensorFlow   suma max|dif| = 0.00e+00   promedio max|dif| = 8.88e-16
  JAX          suma max|dif| = 0.00e+00   promedio max|dif| = 8.88e-16

  Las dos fusiones eligen la MISMA clase en 85.7% de los casos.
```

Ese último número ya adelanta algo: **las dos fusiones no son equivalentes**, y difieren precisamente cuando una rama está mucho más confiada que la otra.

## 3. La curva

| SNR (dB) | solo audio | solo video | fusión tardía | fusión óptima | ganancia (óptima − audio) |
|---|---|---|---|---|---|
| −5 | 12,13 % | **78,42 %** | 79,16 % | 83,49 % | **+71,36** |
| 0 | 27,41 % | 78,41 % | 81,58 % | 89,50 % | +62,08 |
| 5 | 63,84 % | 78,08 % | 88,87 % | 97,02 % | +33,18 |
| 10 | 96,91 % | 78,71 % | 98,70 % | 99,87 % | +2,96 |
| 15 | 99,99 % | 78,11 % | 100,00 % | 100,00 % | +0,01 |
| 20 | 100,00 % | 78,11 % | 100,00 % | 100,00 % | **+0,00** |

Tres propiedades de la figura del paper, reproducidas:

**1. La columna del video es constante** — 78,42 % a −5 dB y 78,11 % a 20 dB. No es que el modelo visual sea robusto: es que **el ruido acústico no lo alcanza**. En la Fig. 3 de Petridis et al. es literalmente una línea horizontal.

**2. A SNR bajo, el video solo supera al audio solo** — 78,42 % contra 12,13 % a −5 dB. En el paper, la línea de V cruza por encima de A y de MFCC alrededor de los −3 dB.

**3. La ganancia de la fusión crece al bajar el SNR** — de +0,00 a 20 dB hasta +71,36 a −5 dB. Petridis et al. miden **+0,3 en limpio y +14,1 a −5 dB**: la misma forma, con magnitudes distintas porque su audio limpio ya está en 97,7 %.

{{< concept-alert type="clave" >}}
La lectura que suele hacerse mal: *"la fusión audiovisual aporta poco (+0,3 puntos)"*. Aporta poco **en condiciones limpias**, y no porque falle sino **porque no hay nada que arreglar** — el audio solo ya está en 97,7 %.

La modalidad débil solo ayuda donde la fuerte falla. Evaluar un sistema multimodal únicamente en condiciones ideales hace invisible su razón de existir.
{{< /concept-alert >}}

## 4. El techo del canal visual

La columna del video no solo es plana: está clavada en ~78 %, y ese techo no se mueve. Midiendo **dónde** caen los errores:

```python
pred = lv.argmax(1)
mismo_par = (pred // 2) == (y // 2)      # ¿acertó al menos el grupo ambiguo?
```

```
exactitud visual exacta        : 78.03%
exactitud 'acierta el par'     : 97.96%
errores que caen DENTRO del par: 90.7%
```

El modelo identifica el grupo correcto el **97,96 %** de las veces; lo que no puede es elegir dentro del grupo. El **90,7 % de sus errores** son confusiones entre las dos clases del mismo par.

{{< concept-alert type="advertencia" >}}
Ese techo **no es un problema de capacidad**. Más datos, más capas o más épocas no lo mueven, porque la información que falta **no está en la entrada**. Es el análogo exacto de los homófonos visuales que LRW incluye deliberadamente en su vocabulario —*America* y *American*— y de los visemas en general.

Distinguir un techo de información de un techo de capacidad es la decisión previa a cualquier trabajo de mejora: uno se resuelve con más modelo, el otro solo con **otra modalidad**.
{{< /concept-alert >}}

## 5. Cómo se fusiona también importa

Volviendo a la tabla de la sección 3, a −5 dB: fusión tardía **79,16 %**, fusión óptima **83,49 %**. Cuatro puntos de diferencia usando exactamente las mismas dos ramas.

La razón es que promediar probabilidades es una operación **aritmética** sobre objetos que viven en escala **logarítmica**. Bajo independencia condicional dada la clase,

$$\log p(c \mid x_a, x_v) = \log p(x_a \mid c) + \log p(x_v \mid c) + \log p(c) + \text{cte}$$

las evidencias se **suman**. Un promedio de probabilidades le da a la rama ruidosa un piso de influencia que la formulación correcta no le concede.

La BiGRU de fusión de Petridis et al. no hace ninguna de las dos cosas: **aprende** la combinación a partir de los datos. Es el argumento de fondo para preferir fusión intermedia sobre fusión tardía cuando hay datos suficientes.

## 6. Cuándo la fusión estorba

Con el audio limpio y el canal visual degradándose progresivamente, mientras el fusor sigue creyendo que ambos valen lo mismo:

```python
lv_bad = clasifica_gauss(Xv_degradado, proto_v, sigma=1.0)   # el modelo CREE que sigma=1
av = np.log(0.5 * softmax(la) + 0.5 * softmax(lv_bad))
```

| $\sigma$ visual real | solo video | solo audio | fusión tardía | delta |
|---|---|---|---|---|
| 1,0 | 77,58 % | 100,00 % | 100,00 % | +0,00 |
| 2,0 | 43,71 % | 100,00 % | 100,00 % | +0,00 |
| 4,0 | 17,03 % | 100,00 % | 100,00 % | +0,00 |
| 8,0 | 7,20 % | 100,00 % | 99,79 % | −0,21 |
| 16,0 | 4,54 % | 100,00 % | **94,51 %** | **−5,49** |

Con el canal visual roto y el peso fijo en 0,5, la fusión queda **por debajo** del audio solo. El promedio a ciegas arrastra a la modalidad buena.

{{< concept-alert type="clave" >}}
Esto le da su verdadero sentido a una línea de la clase que parece rutina de aumentación:

> *Audio: ruido aleatorio agregado a distintos niveles: **[−5 dB, 20 dB]**, base NOISEX.*

Entrenar la BiGRU de fusión sobre **todo el rango de condiciones acústicas** es lo que le permite aprender a ponderar según la calidad de la señal en vez de promediar siempre igual. Si solo hubiera visto audio limpio, habría aprendido a ignorar el video —donde efectivamente no aporta— y habría fallado exactamente en el régimen que justifica el sistema.

**La aumentación con ruido acá no es regularización. Es hacerle ver al fusor el caso que tiene que resolver.**
{{< /concept-alert >}}

---

## Qué se aprendió

1. **La horizontalidad de la línea del video no es robustez del modelo**: es independencia física entre los ruidos de las dos modalidades.
2. **La ganancia de la fusión crece al bajar el SNR**, de +0,00 a +71,36 puntos. En condiciones limpias no aporta porque no hay nada que arreglar.
3. **El canal visual tiene un techo de información**, no de capacidad: el 90,7 % de sus errores son confusiones dentro del par ambiguo.
4. **Sumar log-verosimilitudes supera a promediar probabilidades** por 4,3 puntos a SNR bajo, con las mismas dos ramas.
5. **La fusión con pesos fijos puede perjudicar** (−5,49 puntos) cuando una modalidad se degrada más de lo previsto — y por eso el paper inyecta ruido en todo el rango durante el entrenamiento.

---

**Volver a:** [Práctica](../) · [Profundización](/clases/clase-43/profundizacion) · [Teoría](/clases/clase-43/teoria)
