---
title: "La fuga de folds"
weight: 1
---

El notebook dedica una sección entera —titulada *"Un ligero desvío: K-Fold Cross Validation"*— a explicar por qué el 90 % que los alumnos obtuvieron en el primer laboratorio era mentira:

> *"este resultado es poco fiable por una regla básica que fue violada al entrenar ese modelo en ese momento: el set de entrenamiento no debe estar correlacionado al set de test"*

Y tres celdas más abajo, la clase `AudioDataset` vuelve a violarla de forma más grave: no correlaciona el test con el train, **lo mete adentro**.

## Los corchetes

La línea es esta:

```python
self.audio_paths = glob.glob(audio_paths + '/*' + str(self.folds) + '/*')
```

El argumento `folds` llega como lista. Con `folds = [2,3,4,5,6,7,8,9,10]`, la conversión `str()` produce la cadena literal:

```
"[2, 3, 4, 5, 6, 7, 8, 9, 10]"
```

En `glob`, los corchetes **no delimitan una lista**: definen una **clase de caracteres**, igual que en una expresión regular. El patrón resultante deja de significar "las carpetas 2 a 10" y pasa a significar *"cualquier carpeta cuyo nombre termine en alguno de los caracteres `0123456789`, coma o espacio"*.

`fold1` termina en `1`. Y el carácter `1` está en el conjunto, porque aparece dentro del `10`.

La verificación sobre el dataset descargado:

```
train = 8732 archivos   (el dataset completo, los 10 folds)
test  =  873 archivos   (fold1)
intersección = 873      (el 100 % del test está dentro del train)
```

No es que se filtre *parte* del test. El conjunto de entrenamiento **es el dataset entero**, y cada archivo con el que se mide el rendimiento se vio unas 20 veces durante el entrenamiento.

De paso, la línea revela que la clase nunca soportó listas: con `folds = 5` habría funcionado por accidente, y con `folds = [1]` funciona porque la clase de caracteres de un solo elemento coincide con lo que se pretendía.

## El sesgo no es constante: escala con la capacidad

La corrección es una línea, y permite medir el efecto reentrenando con folds disjuntos y todo lo demás idéntico —arquitectura, semillas, `batch_size`, optimizador, learning rate, scheduler y número de épocas:

```python
class AudioDatasetFold(AudioDataset):
    """Idéntica a AudioDataset salvo por la selección correcta de folds."""
    def __init__(self, file_path, audio_paths, folds):
        super().__init__(file_path, audio_paths, folds)
        folds = folds if isinstance(folds, (list, tuple)) else [folds]
        self.audio_paths = sorted(
            sum([glob.glob(os.path.join(audio_paths, 'fold%d' % f, '*.wav')) for f in folds], []))
```

| Modelo | Params | Notebook (con fuga) | Split corregido | **Efecto de la fuga** | Paper |
|---|---|---|---|---|---|
| M3 | 0.22 M | 56.13 % | 45.13 % | **+11.00** | 56.12 % |
| M5 | 0.56 M | 76.63 % | 52.12 % | **+24.51** | 63.42 % |

Este es el resultado que importa: **la fuga aporta 11 puntos a M3 y 24.5 a M5**. No es un sesgo aditivo que uno pueda restar de todos los números por igual — depende del modelo evaluado. Cuanta más capacidad de memorización tiene la red, más provecho saca de que el conjunto de test esté dentro del de entrenamiento.

La consecuencia práctica es que **la fuga distorsiona también las comparaciones entre arquitecturas**, no solo los niveles absolutos:

| Ganancia por profundidad M3 → M5 | |
|---|---|
| Con fuga | **+20.50 puntos** |
| Split corregido | **+6.99 puntos** |
| Paper (Dai et al. 2016) | **+7.30 puntos** |

Con folds disjuntos, el efecto arquitectónico se reproduce con **0.31 puntos** de diferencia respecto del paper. Con el bug activo, aparecía inflado casi tres veces.

![Curvas de M5 con y sin fuga, y descomposición del rendimiento](/laboratorios/lab-39/fuga-de-datos.jpg)

## Dos errores que se cancelaban

M3 con fuga da **56.13 %**. El paper reporta **56.12 %**. Coincidencia hasta la segunda cifra decimal, con una arquitectura idéntica y un dataset idéntico.

Esa coincidencia no validaba nada. Es la suma de dos sesgos de magnitud casi igual y signo opuesto:

- **+11.00 puntos** de la fuga de folds
- **−10.99 puntos** del preprocesamiento (que se analiza en [la página siguiente](/laboratorios/lab-39/02-el-preprocesamiento))

El déficit contra el paper resulta ser sorprendentemente estable entre modelos —**−10.99 para M3 y −11.30 para M5**, con capacidades y regímenes de sobreajuste muy distintos—, mientras que la fuga escala. En M3 ambos efectos se anulan; en M5 la fuga es más del doble y el resultado se dispara 13 puntos sobre el paper.

{{< concept-alert type="clave" >}}
**Una métrica que coincide con la referencia esperada no valida el pipeline.** Puede estar coincidiendo por cancelación de errores. En este lab, las dos señales que habitualmente se usan para dar por bueno un experimento apuntaban ambas en la dirección equivocada:

1. *"El resultado calza con el paper"* — calzaba por casualidad aritmética.
2. *"Train y test van parejos"* — con el test contenido íntegramente en el train, esa coincidencia es una **tautología**: ambas métricas se calculan sobre el mismo material.

La verificación que sí discrimina es directa y cuesta una línea: comprobar que la intersección entre los conjuntos sea vacía.

```python
assert len(set(train.audio_paths) & set(test.audio_paths)) == 0
```
{{< /concept-alert >}}

## La fuga no solo infla el número: borra el diagnóstico

Lo más costoso del bug no es el sesgo, es lo que oculta. Comparando las dos corridas de M5:

| | Split con fuga | Split corregido |
|---|---|---|
| Train final | 73.69 % | 74.25 % |
| Mejor test | **76.63 %** (época 19) | **52.12 %** (época **4**) |
| Test final | 69.07 % | 49.60 % |
| Gap train − test | **−2.9** (test > train) | **+24.6** |

**El train es casi idéntico en ambos casos: 74.35 % contra 74.25 %.** El modelo aprende exactamente lo mismo; lo único que cambia es sobre qué se mide.

Con el split limpio, la mejor época es la **cuarta**, con el train recién en 56.86 %. A partir de ahí el train sigue subiendo monótono hasta 74.25 % mientras el test se estanca oscilando entre 42 % y 52 %. Eso es sobreajuste de manual —559 K parámetros sobre 7859 ejemplos, sin dropout ni data augmentation— y señala con precisión dónde correspondía detener el entrenamiento.

Con el split roto ese diagnóstico es invisible: las dos curvas suben juntas durante las 20 épocas y el test llega a superar al train. El conjunto de test existe justamente para emitir esa señal, y el bug la destruye.

## La Parte 2 arrastra el mismo bug

El `AudioDataset` que se redefine para VGGish contiene la misma construcción. Se confirma en el contador de entrenamiento: **26196 = 3 × 8732**, o sea el dataset completo multiplicado por los tres parches de 0.96 s en que se corta cada clip.

Ahí el efecto es aún mayor, porque la combinación es la peor posible: embeddings preentrenados muy expresivos, una capa lineal sobre 4096 features y los mismos clips vistos tres veces por época. El detalle está en [la última página](/laboratorios/lab-39/06-agregacion-y-transfer-learning), pero el resumen es que **el 97.14 % que reporta el pipeline completo cae a 84.65 % con folds disjuntos**.

---

**Siguiente:** [El preprocesamiento y sus once puntos](/laboratorios/lab-39/02-el-preprocesamiento) — la otra mitad de la descomposición, y por qué el déficit contra el paper es constante entre modelos.
