---
title: "Ejercicios del Laboratorio"
weight: 40
math: true
---

## Actividades

El laboratorio plantea 3 actividades que requieren modificar la arquitectura de AlexNet. Todas parten del modelo original que trabaja con imagenes de $3 \times 224 \times 224$ y clasifica en 1000 clases (ImageNet).

---

### Actividad 1: Cambiar el numero de clases

**Objetivo:** Alterar el modelo original para que trabaje en clasificacion de **102 clases** en vez de 1000. Nombrar la clase como `MiAlexNet`.

**Pista:** Solo es necesario modificar la ultima capa fully connected (`fc8`), cambiando la dimension de salida de 1000 a 102.

```python
class MiAlexNet(nn.Module):

    def __init__(self):
        super(MiAlexNet, self).__init__()
        # ... mismas capas conv1 a conv5 ...
        self.flat = nn.Flatten()
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
        # CAMBIO: 1000 -> 102
        self.fc8 = nn.Sequential(nn.Linear(4096, ____))

    def forward(self, x):
        # ... mismo forward ...
        return x
```

**Pregunta:** Como cambia la cantidad de parametros respecto al modelo original?

---

### Actividad 2: Cambiar el tamano de entrada

**Objetivo:** Alterar la definicion de MiAlexNet para que tome como input imagenes de $3 \times 64 \times 64$ (en vez de $3 \times 224 \times 224$). Para esto, se deben alterar las capas `conv1` y `conv2` manteniendo el resto iguales.

**Pista:** Con una entrada mas pequena, los hiperparametros de las primeras capas (kernel, stride, padding) deben ajustarse para que las dimensiones sigan siendo compatibles con conv3 en adelante. Hay que recalcular las dimensiones usando la formula:

$$O = \left\lfloor\frac{W - K + 2P}{S}\right\rfloor + 1$$

El objetivo es que la salida de conv2 siga siendo $256 \times 13 \times 13$ (o un tamano compatible con conv3). Despues de conv5, la dimension espacial cambiara, lo que tambien afecta a `fc6`.

```python
# Hay que recalcular:
# 1. conv1: que kernel/stride/padding produce una salida razonable desde 64x64?
# 2. conv2: como llegar a 13x13 para mantener conv3-conv5?
# 3. fc6: cuantas neuronas entran despues de flatten?
```

**Pregunta:** Como cambia la cantidad de parametros? Usar el siguiente codigo para verificar:

```python
def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

modelo = MiAlexNet()
modelo_antiguo = AlexNet()
diferencia = contar_parametros(modelo_antiguo) - contar_parametros(modelo)
print(f"La diferencia es de {diferencia} parametros.")
```

---

### Actividad 3: Agregar una capa convolucional

**Objetivo:** Usando el modelo **original** (sin los cambios de actividades 1 y 2), agregar una capa convolucional despues de `conv5` llamada `conv6` que reduzca la cantidad de filtros a 128. Ademas, alterar las capas lineales para que trabajen con 1024 dimensiones en vez de 4096.

**Detalles:**

1. Agregar `conv6` despues de `conv5`:
   - `in_channels=256` (salida de conv5)
   - `out_channels=128`
   - Elegir kernel, stride y padding apropiados

2. Modificar `fc6`:
   - Recalcular `in_features` basado en la nueva salida despues de conv6 + flatten
   - Cambiar `out_features` a 1024

3. Modificar `fc7`: Linear(1024, 1024)

4. Modificar `fc8`: Linear(1024, 1000)

5. Actualizar la funcion `forward` para incluir `conv6`

```python
class MiAlexNet(nn.Module):

    def __init__(self):
        super(MiAlexNet, self).__init__()
        # ... conv1 a conv5 igual que el original ...

        # NUEVA CAPA
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=(____, ____), stride=(____, ____),
                      padding=(____, ____)),
            nn.ReLU()
        )

        self.flat = nn.Flatten()
        self.fc6 = nn.Sequential(nn.Linear(____, 1024), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())
        self.fc8 = nn.Sequential(nn.Linear(1024, 1000))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)  # nueva capa
        x = self.flat(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
```

**Pregunta:** Como cambia la cantidad de parametros? Dado que las capas lineales ahora usan 1024 en vez de 4096, la reduccion deberia ser significativa.

{{< concept-alert type="clave" >}}
Las tres actividades demuestran un punto importante: la arquitectura de una CNN no es fija. Los hiperparametros (numero de filtros, tamano de kernel, dimensiones de capas lineales) se ajustan segun el problema. Lo critico es mantener la **consistencia dimensional** entre capas consecutivas.
{{< /concept-alert >}}

---

## Parte 2: Aplicacion a clasificacion de flores

Despues de las actividades de arquitectura, el laboratorio aplica MiAlexNet (actividad 1) al dataset **Flowers** de Oxford, que contiene 8,189 imagenes de 102 tipos de flores.

El modelo se carga con pesos preentrenados (el entrenamiento se cubre en laboratorios posteriores) y se evalua su accuracy en train y test para observar el sobreajuste tipico de redes profundas entrenadas con datasets pequenos.
