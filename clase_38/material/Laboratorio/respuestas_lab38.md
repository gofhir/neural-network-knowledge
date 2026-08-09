# Celdas markdown para el Lab 38 — copiar/pegar en Colab

---

## BLOQUE A — Hallazgos experimentales (celda de texto después de la 46)

### Hallazgos experimentales

Al ejecutar la actividad, el modelo predijo **`rock climbing` (87,66 %)** en lugar de `abseiling`
(11,23 %, rank #2). Diagnostiqué la causa con cinco experimentos controlados sobre el mismo video:

| Hipótesis | Resultado | Conclusión |
|---|---|---|
| **H1** Rango de entrada `[0,1]` vs `[-1,1]` | 10,86 % → **62,31 %**; el top-1 cambia | **Causa principal** |
| **H2** `crop256 + resize224` vs `crop224` directo | +2,9 pts | Marginal |
| **H3** Crop izquierdo / centro / derecho | 9,3 % / 65,2 % / **75,3 %**; entropía 2,75 en el izquierdo | El sujeto no está centrado |
| **H4** Ventana temporal | primeros 64 frames: **92,9 %**; últimos 100: 39,4 % (falla) | Gradiente temporal fuerte |
| **H5** Video invertido en el tiempo | 65,2 % → 67,2 % (sin cambio) | I3D **no** usa la dirección del tiempo |

**Causa raíz (H1).** El notebook normaliza a `[0,1]` con `np.array(frames) / 255.0`, pero el
repositorio oficial de DeepMind (`google-deepmind/kinetics-i3d`) especifica *"pixel values are then
rescaled between -1 and 1"*. I3D hereda las capas BatchNorm de Inception-v1, calibradas sobre
entradas centradas en cero; alimentarlo con `[0,1]` introduce un sesgo constante de +0,5 que se
propaga por toda la red.

El error **no destruye el modelo**, y por eso puede pasar desapercibido: `archery` se clasifica
correctamente con ambos preprocesos (99,66 % con `[0,1]` vs 99,97 % con `[-1,1]`). Sólo se
manifiesta en la frontera de decisión entre clases visualmente vecinas.

| Video | Régimen | `[0,1]` | `[-1,1]` | Efecto del preproceso |
|---|---|---|---|---|
| `archery` | en vocabulario, fácil | 99,66 % (correcto) | 99,97 % (correcto) | marginal |
| `abseiling` | en vocabulario, grano fino | 10,86 % (rank #2) | 62–93 % (rank #1) | **invierte la predicción** |
| `ApplyEyeMakeup` | fuera de vocabulario | 98,23 % (H≈0,11) | 79,50 % (H≈0,55) | mejora la calibración, no la respuesta |

**Validación externa.** La demo oficial de GluonCV usa este mismo video y reporta `abseiling` con
99,1 %, empleando 32 frames de la ventana inicial y crop central de 224 px — coherente con H1 y H4.

**Sobre el 3-crop.** Promediar los logits de los tres crops horizontales **empeoró** el resultado
(75,5 % frente a 92,9 % usando sólo el crop central), porque el crop izquierdo no es un votante
neutro: vota activamente por `rock climbing`. El protocolo estándar de múltiples crops supone que
el sujeto está aproximadamente centrado; en este clip de 455×256 está desplazado a la derecha.

**Sobre H5.** Invertir temporalmente el video no cambia la predicción, pese a que un rápel invertido
es visualmente una escalada. I3D discrimina por apariencia y movimiento de corto alcance (la cuerda,
el arnés, la postura), no por la dirección del movimiento. Es consistente con el sesgo de apariencia
documentado en Kinetics-400 y con la motivación del dataset Something-Something.

**Sobre la entropía como señal de alerta.** La entropía detectó la ambigüedad de `abseiling`
(0,416 nats frente a 0,064 de `archery`) y el crop sin sujeto (2,753 nats), pero fue **ciega** al
caso de `ApplyEyeMakeup` (0,106 nats con una clase que ni siquiera existe entre las 400). Mide
ambigüedad entre clases conocidas y falta de evidencia, no ausencia de vocabulario.

---

## BLOQUE B — Respuesta 1 (celda 48)

### 1. ¿El modelo I3D es un modelo 3D o 2D?

**Respuesta 1:**

I3D es un modelo **3D**: todas sus capas son convoluciones y poolings tridimensionales, con kernels
de forma `t × k × k` que se deslizan simultáneamente sobre las dimensiones de tiempo, alto y ancho.
Su entrada es un tensor 5D `[batch, frames, alto, ancho, canales]` — en este práctico,
`(1, 250, 224, 224, 3)` — algo que una CNN 2D no puede procesar sin destruir el eje temporal.

Sin embargo, **su origen es 2D**, y de ahí viene su nombre: la "I" es de *Inflated*. Carreira y
Zisserman no diseñaron una arquitectura 3D nueva, sino que tomaron una **Inception-v1 ya entrenada
en ImageNet** e "inflaron" cada filtro: un kernel 2D de `k × k` se replica `t` veces a lo largo del
eje temporal y se **divide por `t`**. Esa división garantiza que, ante un video de imagen congelada
(el mismo frame repetido), la red 3D produzca activaciones numéricamente idénticas a las de la red
2D original — lo que el paper llama el *boring video fixed point*. La red inflada arranca
reproduciendo exactamente una CNN de ImageNet y sólo debe aprender la componente de movimiento.

Encontré **evidencia experimental directa de esa herencia 2D**. Al comparar rangos de entrada, el
video de rápel se clasifica como `rock climbing` (88,02 %) si se normaliza a `[0,1]` —como hace el
notebook—, pero la predicción se invierte a `abseiling` (62,31 %) al normalizar a `[-1,1]`, que es
el rango que especifica el repositorio oficial de DeepMind. La explicación es que I3D **hereda las
capas BatchNorm de Inception-v1**, cuyas estadísticas fueron calibradas con entradas centradas en
cero. El modelo es tan literalmente una red 2D inflada que arrastra incluso las estadísticas de
normalización de su versión para imágenes.

En resumen: I3D es **3D en su funcionamiento y 2D en su inicialización**, y es esa combinación la
que le permite heredar el conocimiento visual de ImageNet en vez de entrenar desde cero.

*(Nota: el I3D completo del paper es además two-stream — una rama RGB y otra de flujo óptico, ambas
3D, cuyos logits se promedian. El módulo de TensorFlow Hub usado en este práctico es sólo la rama RGB.)*

---

## BLOQUE C — Respuesta 2 (celda 49)

### 2. ¿El modelo I3D es un enfoque de modelo recortado (trimmed model approach)?

**Respuesta 2:**

Sí, I3D es un enfoque **trimmed** (de video recortado). La distinción relevante es:

- **Trimmed:** clips cortos (~10 s) ya recortados de modo que la acción ocupa todo el clip. La tarea
  es de **clasificación**: un clip produce una etiqueta. Ejemplos: Kinetics, UCF101, HMDB51.
- **Untrimmed:** videos largos en los que la acción ocupa un segmento temporal desconocido. La tarea
  es de **detección o localización temporal**: hay que decir qué acción ocurre y entre qué instantes.
  Ejemplos: THUMOS14, ActivityNet.

I3D pertenece al primer grupo, por tres razones verificables en este mismo práctico:

**1. Su salida no tiene eje temporal.** El modelo devuelve un tensor `[1, 400]`: un único vector de
clases para el clip completo, porque aplica *average pooling* sobre el tiempo antes de la capa de
clasificación. No puede indicar **cuándo** ocurre la acción, sólo **qué** acción hay, asumiendo que
hay una y que dura todo el clip.

**2. Fue entrenado y evaluado sobre datasets recortados.** Kinetics-400 está formado por clips de
~10 s recortados manualmente; UCF101 y HMDB51 también son recortados. El video de la actividad dura
exactamente 10,00 s a 25 fps: los parámetros canónicos de un clip de Kinetics.

**3. Medí el costo de esa suposición.** Evaluando distintas ventanas temporales del mismo video obtuve:

| Ventana | Frames | p(`abseiling`) | Top-1 |
|---|---|---|---|
| Primeros 64 (0–2,6 s) | 64 | **92,85 %** | `abseiling` |
| Centrales 100 (3–7 s) | 100 | 58,46 % | `abseiling` |
| Últimos 100 (6–10 s) | 100 | 39,36 % | `rock climbing` (falla) |
| Video completo | 250 | 65,19 % | `abseiling` |

Existe un gradiente temporal claro: el inicio del clip es inequívocamente rápel y el final se parece
a una escalada. Promediar sobre los 10 s completos **diluye la evidencia buena del inicio con la
evidencia ambigua del final**, y como el promediado ocurre sobre los logits dentro de la red, esa
información ya no se puede recuperar.

Esto ilustra exactamente la limitación *trimmed*: I3D supone que toda la ventana que recibe contiene
la acción. Cuando no es así, no se equivoca "un poco" — puede invertir la predicción. Para video
untrimmed, I3D se emplea entonces como **extractor de características** sobre ventanas deslizantes,
con una cabeza adicional encargada de la localización temporal.

---

## BLOQUE D — Respuesta 3 (celda 50)

### 3. Menciona al menos una ventaja del I3D sobre los enfoques anteriores

**Respuesta 3:**

**Ventaja principal: el inflado resuelve el cuello de botella de datos.**

Una CNN 3D tiene muchos más parámetros que su equivalente 2D, porque cada kernel se multiplica por
el tamaño temporal; pero los datasets de video eran órdenes de magnitud menores que ImageNet. C3D
(Tran, 2015) tenía **78 millones de parámetros entrenados desde cero** sobre los ~13.000 videos de
UCF101, con el sobreajuste garantizado que eso implica.

El inflado rompe ese círculo: I3D **hereda los pesos de ImageNet** y opera con unos **12 millones de
parámetros — 6,5 veces menos que C3D —** obteniendo resultados muy superiores (**98,0 %** en UCF101
frente a ~88 % del estado del arte previo, y **80,9 %** en HMDB51 frente a ~67 %). No hacen falta
millones de videos anotados porque las primeras capas —bordes, texturas, partes de objetos— ya se
aprendieron con imágenes fijas; la red sólo debe aprender la componente temporal. Como beneficio
adicional, al ser el inflado una operación mecánica aplicable a cualquier CNN 2D, la arquitectura
hereda gratis todo el progreso futuro en diseño de redes de imagen, algo imposible para C3D.

**Ventaja adicional: representaciones transferibles de alta calidad.**

En este práctico, sin ningún entrenamiento, el modelo pre-entrenado clasificó `archery.mp4` con
**99,97 %** de confianza. Sus cuatro competidores en el top-5 —`throwing axe`, `flying kite`,
`catching or throwing frisbee` y `pole vault`— comparten el mismo patrón cinemático de brazos
extendidos, objeto alargado y gesto de lanzar. El espacio de representación agrupa acciones por
**estructura del movimiento**, no por contexto ni por objeto, que es justamente lo que se espera de
filtros espacio-temporales aprendidos end-to-end. Los enfoques previos no lo lograban: la CNN 2D con
pooling (Karpathy, 2014) ignora el orden de los frames; Two-Stream (Simonyan, 2014) precomputa el
flujo óptico **fuera** de la red, de forma cara y no aprendible; y LRCN (Donahue, 2015) colapsa cada
frame a un vector antes de que el LSTM lo procese, perdiendo el movimiento local.

**Dos matices que también encontré experimentalmente.**

Sobre `ApplyEyeMakeup`, cuya clase **no existe** en Kinetics-400, el modelo predijo `filling
eyebrows` con **98,23 %** de confianza. Un modelo pre-entrenado sólo puede responder dentro de su
vocabulario, y el softmax le impide expresar "no lo sé". Por eso el propio paper *Quo Vadis*
reemplaza la capa final de 400 salidas por una de 101 y hace fine-tuning para alcanzar el 98,0 % en
UCF101: el pre-entrenamiento entrega un extractor de características excelente, no un clasificador
directamente utilizable.

Al invertir temporalmente el video de rápel, la predicción prácticamente no cambió (65,19 % → 67,22 %
para `abseiling`), pese a que un rápel invertido es visualmente una escalada. Esto sugiere que I3D
discrimina estas dos clases por apariencia y movimiento de corto alcance —la cuerda, el arnés, la
postura— más que por la dirección del movimiento. Es coherente con el sesgo de apariencia
documentado en Kinetics-400 y con la motivación del dataset Something-Something. La arquitectura 3D
**habilita** el razonamiento temporal, pero el dataset no siempre lo **exige**.
