---
title: "El learning rate decide el orden"
weight: 4
---

Si la [tesis del campo receptivo](/laboratorios/lab-39/03-familia-m-y-campo-receptivo) es correcta, M18 —con 1358 ms de contexto y 3.7 M de parámetros— debería superar a M5. El paper lo confirma: 71.68 % contra 63.42 %.

El notebook produce lo contrario:

| Modelo | Test | Train final |
|---|---|---|
| M5 | **76.63 %** | 73.69 % |
| M18 | **60.82 %** | 60.17 % |

M18 queda **15.8 puntos por debajo**. Si uno se detuviera acá, la conclusión sería que la profundidad no ayuda sobre este dataset.

## M18 no sobreajusta: está subentrenado

La clave no está en el test sino en el train.

| Modelo | Params | Train final | Train en la época 1 |
|---|---|---|---|
| M3 | 0.22 M | 56.14 % | 31.02 % |
| M5 | 0.56 M | **73.69 %** | 31.80 % |
| M18 | **3.7 M** | **60.17 %** | **20.16 %** |

**M18 tiene 6.6 veces más parámetros que M5 y ajusta peor el conjunto de entrenamiento.** Eso descarta el sobreajuste: un modelo que sobreajusta memoriza el train, y este ni siquiera llega a donde llegó M5. Está a mitad de camino de converger.

![Curvas de entrenamiento de M18, 20 épocas](/laboratorios/lab-39/m18-curvas.jpg)

Tres señales lo confirman, y son la respuesta a la **Actividad 4**:

**Arranque lento.** M3 y M5 alcanzan ~31 % en la primera época; M18 se queda en 20.16 % y necesita tres épocas para llegar ahí. Y la pérdida **sube** entre la primera y la segunda época (2.077 → 2.297): los primeros pasos son inestables.

**La curva de train no satura.** M3 se aplana al final (55.49 → 55.55 → 56.14), mientras M18 sigue subiendo sin techo (58.04 → 58.82 → 59.59 → 60.17). El entrenamiento se cortó en plena pendiente.

**El test oscila el doble.** M3 y M5 varían ±10 puntos entre épocas consecutivas; M18 salta de **59.11 % a 34.48 % en una sola época**, y luego a 50.86 %. Un rango de 26 puntos en las últimas diez épocas, contra los 14 de M3.

## El perfil de errores cambia por completo

Aunque M18 pierde en accuracy —y su F1 es menor que el de M5 en nueve de las diez clases, con un empate en `engine_idling`—, el **carácter** de sus errores es distinto y revela algo que el número global esconde:

| Clase | M5 (P / R / ratio) | M18 (P / R / ratio) |
|---|---|---|
| **gun_shot** | 0.58 / 0.94 / **1.63×** | **1.00** / 0.43 / **0.43×** |
| **siren** | 0.62 / 0.87 / **1.41×** | **0.91** / 0.45 / **0.50×** |
| car_horn | 0.81 / 0.61 / 0.75× | 0.60 / **0.17** / **0.28×** |
| air_conditioner | 0.96 / 0.72 / 0.75× | 0.35 / 0.48 / **1.37×** |
| street_music | 0.79 / 0.56 / 0.71× | 0.54 / **0.70** / **1.30×** |

![Matriz de confusión de M18](/laboratorios/lab-39/m18-matriz.jpg)

**`gun_shot` alcanza precision 1.00.** El modelo lo predice 15 veces y acierta las 15. En M3 era el peor sumidero del clasificador (1.86× de sobrepredicción, precision 0.37) y en M5 seguía siéndolo (1.63×). Con M18 se invierte por completo.

Es la evidencia más limpia a favor de la tesis del campo receptivo. Lo que distingue un disparo de un golpe de martillo neumático no es el ataque —ambos son transitorios impulsivos de banda ancha, indistinguibles en 19.5 ms— sino **la cola: el patrón de decaimiento y reverberación**, que se despliega en cientos de milisegundos. M18 puede verla, y cuando la ve no se equivoca. `siren` sigue el mismo patrón (precision 0.91): el barrido lento de frecuencia tiene un período de uno o dos segundos.

**El precio es el recall**: `gun_shot` 0.43, `siren` 0.45, `car_horn` 0.17. Es la firma típica de un modelo subentrenado — aprendió primero los ejemplos prototípicos con reglas de altísima confianza, y aún no cubre la variabilidad de cada clase. La masa que no reconoce cae en las clases genéricas, que pasan a ser los nuevos sumideros.

`car_horn` colapsa a recall 0.17: 6 aciertos de 36, con el modelo atreviéndose a predecirla solo 10 veces. Es la clase más pequeña junto con `gun_shot` y la de evento más breve. **Cuando el presupuesto de entrenamiento no alcanza, la clase minoritaria y difícil es la primera en sacrificarse.**

## El experimento: bajar el learning rate

`lr = 0.01` es diez veces el valor por defecto de Adam. Repitiendo la corrida con `lr = 0.001` y sin cambiar nada más:

| Modelo | `lr = 0.01` | `lr = 0.001` | ganancia |
|---|---|---|---|
| M18 | 60.82 % | **83.85 %** | **+23.02** |

**+23 puntos por un hiperparámetro.** Y el orden se restaura:

| | `lr = 0.01` | `lr = 0.001` | Paper |
|---|---|---|---|
| M3 | 56.13 % | — | 56.12 % |
| M5 | 76.63 % | 81.10 % | 63.42 % |
| M18 | **60.82 %** ❌ | **83.85 %** ✅ | 71.68 % |
| Orden M5 vs M18 | M18 −15.8 | **M18 +2.75** | M18 +8.26 |

M18 con `lr = 0.001` ni siquiera había convergido: su mejor resultado es la **época 20**, la última, con el train todavía subiendo (76.88 %).

## El control que hace válida la conclusión

Con lo anterior todavía no se puede afirmar que el learning rate perjudique **específicamente a las redes profundas**. Si M5 también ganara 23 puntos con el mismo cambio, el efecto no tendría nada que ver con la profundidad: sería simplemente que 0.001 va mejor para todos.

| Modelo | capas | params | `lr = 0.01` | `lr = 0.001` | **ganancia** |
|---|---|---|---|---|---|
| M5 | 5 | 0.56 M | 76.63 % | 81.10 % | **+4.47** |
| M18 | 18 | 3.68 M | 60.82 % | 83.85 % | **+23.02** |

M18 gana **5.2 veces más** que M5 con la misma modificación. El efecto escala con la profundidad.

{{< concept-alert type="clave" >}}
**La interacción invierte el signo del efecto.** Con `ReLU` sobre los logits —la configuración de la Parte 2— conviene `lr = 0.001`; sin él, conviene `lr = 0.01`. Y acá, con M5 conviene poco cambiar y con M18 conviene mucho.

Eso es exactamente por qué se diseña un experimento factorial en lugar de dos pruebas sueltas: **variando un factor a la vez con el otro fijo se puede llegar a la conclusión opuesta a la correcta**. Comparar arquitecturas con hiperparámetros congelados no es una comparación neutral — el learning rate óptimo no es independiente de la profundidad de la red.
{{< /concept-alert >}}

## Por qué la profundidad reduce el learning rate utilizable

Esta es la respuesta a la **Actividad 5**, y son tres mecanismos que se suman:

**Composición del efecto a lo largo de las capas.** Un desplazamiento de los pesos en una capa temprana modifica la distribución de las activaciones que reciben todas las capas posteriores. En una red de 3 capas ese efecto se propaga dos veces; en una de 18, diecisiete. Con pasos grandes, cada actualización reubica las representaciones intermedias más rápido de lo que las capas siguientes logran adaptarse.

**Estadísticas de BatchNorm desfasadas.** M18 tiene 18 capas de normalización, cada una con medias y varianzas móviles que en modo `eval` **reemplazan** a las del batch. Si los pesos se mueven mucho entre épocas, esas estadísticas quedan desactualizadas respecto de los pesos actuales. Es la explicación de que la inestabilidad se manifieste sobre todo en la métrica de **test** —medida en `eval`— mientras la de **train** —medida con estadísticas del batch— avanza suave. Y encaja con que la propia clase 39 destaque la BatchNorm como lo que hace entrenables a estas redes: el mismo mecanismo que las habilita es el que las vuelve sensibles al tamaño del paso.

**Presupuesto de entrenamiento insuficiente.** Dai et al. entrenan entre 100 y 400 épocas *hasta convergencia*, y reportan que M18 cuesta 98 s por época contra 63 s de M5. La ventaja de la profundidad **solo se cobra después de converger**; antes de eso, la capacidad adicional es un costo, porque hay más parámetros que ajustar con el mismo número de pasos.

Dicho de otro modo: la profundidad no es gratis, exige un presupuesto de entrenamiento proporcional. El mismo paper que promueve redes profundas sobre onda cruda las entrena entre 5 y 20 veces más épocas de las que el práctico asigna.

---

**Siguiente:** [El ReLU sobre los logits](/laboratorios/lab-39/05-el-relu-sobre-los-logits) — la Parte 2 y su defecto más caro.
