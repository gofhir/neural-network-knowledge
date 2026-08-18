---
title: "Separación de Fuentes"
weight: 142
math: true
---

La **separación de fuentes** descompone una mezcla en las señales que la componen: aislar una voz entre varias, separar los instrumentos de una canción, extraer el habla del ruido de fondo. Es el problema que la psicoacústica llamó *cocktail party* — la capacidad humana de seguir una conversación en una sala llena de gente hablando.

Este fundamento acompaña a la [Clase 44](/clases/clase-44), donde aparece en dos de las siete aplicaciones que se presentan.

---

## 1. El problema

Se observa una mezcla $y(t) = \sum_{i=1}^{S} x_i(t)$ y se quieren recuperar las $S$ fuentes. Con un solo canal de grabación, el problema está **subdeterminado**: una ecuación, $S$ incógnitas. No hay solución sin supuestos adicionales sobre cómo son las fuentes.

El enfoque dominante trabaja en el dominio tiempo-frecuencia. Se calcula la [STFT](/fundamentos/representacion-tiempo-frecuencia) de la mezcla y se estima una **máscara** por fuente:

$$\hat{X}_i(t,f) = M_i(t,f)\cdot Y(t,f)$$

y luego se vuelve al dominio temporal. La razón de fondo es la **dispersión** del habla en ese dominio: en la mayoría de las celdas tiempo-frecuencia domina una sola fuente, así que una máscara binaria ya recupera bastante. Las máscaras suaves (ratio masks) y las complejas —que también corrigen la fase— rinden mejor.

## 2. El problema de la permutación

El obstáculo específico que hizo difícil este problema durante años, y que la literatura llama *label permutation problem*.

Una red que produce $S$ salidas no tiene forma de saber **en qué orden** debe ponerlas. Si en el ejemplo de entrenamiento la salida 1 debe ser la voz grave y la 2 la aguda, pero en el siguiente ejemplo el orden natural es el inverso, la pérdida castiga a la red por una decisión arbitraria — y los gradientes se cancelan entre ejemplos.

Las soluciones conocidas:

- **Deep clustering**: en vez de predecir máscaras, aprender un *embedding* por celda tiempo-frecuencia y agrupar; el agrupamiento no tiene orden, así que el problema desaparece.
- **Permutation invariant training (PIT)**: probar todas las asignaciones posibles entre salidas y referencias, y quedarse con la de menor pérdida.
- **Condicionar por la fuente**: si la red recibe una indicación de *cuál* fuente extraer, no hay nada que ordenar.

{{< concept-alert type="clave" >}}
La tercera es la que hace interesante al caso audiovisual. En [Looking to Listen](/papers/looking-to-listen-ephrat-2018), la **cara de cada hablante** es la condición: la salida $i$ es, por construcción, la voz de la persona cuyo rostro se pasó como entrada. El problema de la permutación no se resuelve — **se disuelve**, porque el video ya especifica el orden.

Es el mismo patrón que atraviesa la [Clase 43](/clases/clase-43) y la 44: la modalidad visual no aporta información acústica, aporta **una estructura de la que el audio carece**.
{{< /concept-alert >}}

## 3. Guiar la separación con video

Dos formas de aprovechar la imagen:

**Por identidad del hablante.** El movimiento de los labios de una persona está correlacionado con su voz y con nada más. [Looking to Listen](/papers/looking-to-listen-ephrat-2018) (Google, 2018) extrae *embeddings* faciales por cuadro, los combina con el espectrograma de la mezcla mediante convoluciones dilatadas y una BiLSTM, y produce una máscara por rostro detectado. Se entrenó sobre **AVSpeech**, un dataset construido a partir de 290 000 charlas y conferencias de YouTube, filtradas para conservar solo segmentos con un único hablante visible y audio limpio — que después se **mezclan sintéticamente** para fabricar los pares de entrenamiento.

**Por objeto sonoro.** [Learning to Separate Object Sounds](/papers/separating-object-sounds-gao-2018) (2018) va más allá del habla: aprende, a partir de video sin etiquetar, qué objeto produce qué componente del sonido, combinando una red visual con **factorización de matrices no negativas** sobre el espectrograma.

{{< concept-alert type="recordar" >}}
La generación de datos de entrenamiento por **mezcla sintética** es la práctica estándar del área y tiene un sesgo que conviene tener presente: las mezclas artificiales no reproducen la acústica real de una sala —reverberación, reflexiones, movimiento de las personas, efecto Lombard (la gente habla más fuerte cuando hay ruido)—. Un modelo entrenado solo con sumas de grabaciones limpias suele degradarse al enfrentar grabaciones reales de varios micrófonos en un ambiente vivo.
{{< /concept-alert >}}

## 4. Cómo se evalúa

Las métricas clásicas descomponen el error en tres partes: **SDR** (relación señal-distorsión, la medida global), **SIR** (cuánta señal de las otras fuentes quedó) y **SAR** (cuánto artefacto introdujo el propio algoritmo). Suele reportarse la **mejora** sobre la mezcla de entrada (SDRi).

La advertencia habitual: SDR alto no garantiza que el resultado suene bien. Un método puede eliminar la interferencia introduciendo artefactos metálicos muy audibles, o dejar interferencia residual que resulta perceptualmente inofensiva. Como en [super-resolución](/fundamentos/super-resolucion), las métricas de distorsión y las de calidad perceptual no coinciden.

## 5. Aplicaciones

Audífonos y prótesis auditivas que aíslan al interlocutor; preprocesamiento para [reconocimiento de voz](/fundamentos/reconocimiento-de-voz) en ambientes ruidosos; producción musical (separación de *stems*); accesibilidad en videoconferencia; y análisis forense de audio — con la misma advertencia que en super-resolución: lo que el modelo entrega es su mejor reconstrucción bajo un prior, no la señal original.

---

## Ver también

- [Looking to Listen (2018)](/papers/looking-to-listen-ephrat-2018) — la separación guiada por rostros.
- [Learning to Separate Object Sounds (2018)](/papers/separating-object-sounds-gao-2018) — separación por objeto visual.
- [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) — el marco general.
- [Representación Tiempo-Frecuencia](/fundamentos/representacion-tiempo-frecuencia) — el dominio donde se opera.
- [Clase 44](/clases/clase-44) · [Clase 43](/clases/clase-43)
