---
title: "06 - Los defectos del notebook"
weight: 60
math: true
---

> Un ampersand sin comillas que manda `wget` a segundo plano y hace fallar el `unzip` de la línea siguiente, un `unzip` que cuelga la celda esperando entrada que nunca llega, un `nn.Parameter` sin inicializar que puede traer NaN de forma no reproducible, un `if` sin `else` que convierte una descarga fallida en veinte minutos de números sin significado, y un `print` que generó 37.720 líneas de las que Colab tiró 32.721.

---

## 1. Los ampersands de la celda 15

```python
!if [ ! -f vox1_test_wav.zip ]; then wget -q --show-progress -O vox1_test_wav.zip https://www.dropbox.com/scl/fi/ue04a6jhyotw1fy2m4k1p/vox1_test_wav.zip?rlkey=n7ogao1unhdxxy19g3jan8tc4&st=f73twwex&dl=0; fi
!unzip -q vox1_test_wav.zip
```

La URL **no está entrecomillada**, y en bash `&` es un **operador de control**: pone el comando anterior en segundo plano. El trace del shell lo muestra:

```
+ '[' '!' -f NOPE.zip ']'
+ echo wget -q --show-progress -O vox1_test_wav.zip \
       'https://www.dropbox.com/scl/fi/ue04a6jhyotw1fy2m4k1p/vox1_test_wav.zip?rlkey=n7ogao1unhdxxy19g3jan8tc4'
+ dl=0
+ st=f73twwex
```

Bash lee la línea como **tres comandos**:

1. `wget … "…?rlkey=n7ogao1unhdxxy19g3jan8tc4"` **&** ← la URL se corta y el proceso va a **segundo plano**
2. `st=f73twwex` **&** ← asignación de variable, también en segundo plano
3. `dl=0` ← otra asignación

### Los dos efectos, uno inofensivo y uno fatal

**(a) La URL llega truncada — y da igual.** Se pierden `&st=` y `&dl=0`. Verificado con peticiones HTTP: la URL truncada devuelve `content-type: application/zip` y sus primeros bytes son `PK\x03\x04` seguidos de `wav/UT`. Y `dl=0` y `dl=1` sirven **exactamente el mismo binario** cuando el cliente es `wget`/`curl` (el parámetro `dl` solo cambia el comportamiento en un navegador).

**(b) `wget` corre en segundo plano — y esto rompe todo.** La línea retorna de inmediato, sin esperar la descarga de **1.023 MB**, y entonces se ejecuta la línea siguiente sobre un archivo que tiene ~0 bytes:

```
End-of-central-directory signature not found. … vox1_test_wav.zip may be a plain
executable file, or it may be incorrectly truncated
```

{{< concept-alert type="advertencia" >}}
**Por qué el bug sobrevivió en el material del curso:** al volver a ejecutar la celda, el `if [ ! -f … ]` ve el archivo (ya completo) y no re-descarga, así que el `unzip` funciona. **El bug se «arregla» re-ejecutando**, y nadie investiga.

El riesgo real es re-ejecutar *antes* de que termine la descarga: `unzip` procesa un zip **parcial**, puede extraer con éxito unos miles de archivos y fallar al final. Entonces el fallo aparece veinte minutos después, en la extracción de features, como un `FileNotFoundError` sobre algún audio — o peor, como un EER extrañamente malo si el archivo faltante no revienta.
{{< /concept-alert >}}

---

## 2. El `unzip` que cuelga la celda

```python
!unzip -q vox1_test_wav.zip
```

Sin guardia y sin política de sobreescritura. La primera vez funciona. **La segunda**, con `wav/` ya poblado, `unzip` pregunta por cada archivo:

```
replace wav/id10270/x6uYqmx31kE/00001.wav? [y]es, [n]o, [A]ll, [N]one, [r]ename:
```

y espera entrada por `stdin`, que en Colab **nunca llega**. La celda queda colgada indefinidamente. Es el síntoma más molesto del notebook, y la causa es la ausencia de un flag: `-n` (*never overwrite*, salta los existentes) o `-o` (sobreescribe sin preguntar, re-escribiendo 1 GB).

### El reemplazo

```python
%%bash
set -e   # aborta si algo falla, en vez de seguir en silencio

if [ ! -f torch_weights.h5 ]; then
  wget -q --show-progress "https://www.dropbox.com/s/u4ptztlgj3n9rot/torch_weights.h5"
fi
if [ ! -f voxceleb1_veri_test.txt ]; then
  wget -q --show-progress "https://www.dropbox.com/s/fqf0ho04eyzmrrs/voxceleb1_veri_test.txt"
fi
if [ ! -f vox1_test_wav.zip ]; then
  wget -q --show-progress -O vox1_test_wav.zip \
    "https://www.dropbox.com/scl/fi/ue04a6jhyotw1fy2m4k1p/vox1_test_wav.zip?rlkey=n7ogao1unhdxxy19g3jan8tc4&st=f73twwex&dl=1"
fi
if [ ! -d wav ]; then
  unzip -q -n vox1_test_wav.zip
fi

echo "--- verificacion ---"
ls -l torch_weights.h5 voxceleb1_veri_test.txt vox1_test_wav.zip
echo "audios extraidos: $(find wav -name '*.wav' | wc -l)"
echo "hablantes        : $(ls wav | wc -l)"
```

Cuatro cambios: **comillas** en las URLs (mata el bug del `&`), **`%%bash`** con `set -e` (una sola shell, aborta al primer fallo), **`-n`** en `unzip`, y **guardia + verificación**.

Los tamaños a comprobar, verificados contra el servidor:

| Archivo | Bytes exactos | |
|---|---|---|
| `torch_weights.h5` | **48.474.142** | 46,2 MB |
| `voxceleb1_veri_test.txt` | **2.338.640** | 2,2 MB |
| `vox1_test_wav.zip` | **1.072.793.438** | 1.023 MB |
| audios extraídos | **4.874** | de 40 hablantes |

De esos 4.874 audios, el protocolo usa **4.715** (los que aparecen en la lista de pares).

---

## 3. El `nn.Parameter` sin inicializar

```python
self.cluster = nn.Parameter( torch.Tensor( self.k_centers + self.g_centers, d_size ) )
```

Llamar a `torch.Tensor` con enteros **no crea un tensor con esos valores: reserva memoria sin inicializarla.** Es equivalente a `torch.empty(10, 512)`. Los 5.120 números son lo que hubiera en ese bloque de memoria.

Y el resultado es peor que «valores aleatorios». En un proceso:

```
intento 0: min=+nan max=+nan nan=179 inf=0 |media|=nan
intento 1: min=+nan max=+nan nan=179 inf=0 |media|=nan
intento 2: min=+nan max=+nan nan=179 inf=0 |media|=nan
```

**179 NaN de 5.120.** Y en otro proceso, donde antes se construyó el modelo completo:

```
cluster contiene 0 NaN de 5120 antes de cargar nada
```

**Cero.** El contenido depende de qué había en el heap. **El comportamiento no es reproducible entre ejecuciones** — la peor propiedad posible de un bug.

Normalmente no se nota, porque `load_state_dict` sobreescribe `cluster` antes de usarlo. El problema es cuando no lo hace.

### Los dos modos de fallo

**Caso 1 — la basura incluye NaN.** El NaN es absorbente: contamina el embedding, los scores, y entonces:

```
(nan > 0.5) = 0.0   -> todo par se clasifica como NEGATIVO
```

En IEEE 754, **toda** comparación con NaN es falsa. `roc_curve` recibiría una constante y el EER que saliera sería un artefacto del interpolador.

**Caso 2 — la basura son números grandes.** Más traicionero, porque no hay NaN y nada parece roto:

| ‖c_k‖ | coseno entre audios **distintos** | |
|---|---|---|
| 0,1 | 0,8369 | discrimina |
| 1,0 | 0,8374 | discrimina |
| 10 | 0,8809 | discrimina |
| **100** | **0,9959** | **colapsado** |

La razón es algebraica: en `v_k = Σ a_ik (x_i − c_k)`, si `‖c_k‖ ≫ ‖x_i‖` domina el término `−c_k Σ a_ik`. Entonces `v_k ≈ −c_k · (masa)`, y al intra-normalizar queda `≈ −c_k/‖c_k‖`: **una dirección fija que no depende del audio.** Todos los embeddings colapsan, el coseno tiende a 1 y el EER se va a ~50 %: azar puro.

**El arreglo, una línea:**

```python
self.cluster = nn.Parameter( torch.randn( self.k_centers + self.g_centers, d_size ) * 0.01 )
```

---

## 4. El `if` sin `else`: el fallo silencioso

```python
network_eval = NetVlad()                       # centroides = basura sin inicializar

if os.path.isfile( weights_file ):             # ← sin else
    print( 'loading weights from file [%s]' % (weights_file) )
    network_eval.load_state_dict( ... )
network_eval = network_eval.to( device )        # ← se ejecuta pase lo que pase
```

Si `torch_weights.h5` no existe —descarga fallida, directorio equivocado, un `%cd` de más— **no pasa nada visible**: no hay excepción, no hay warning, ni siquiera el `print`. El modelo sigue con 12 M de pesos aleatorios y 5.120 centroides de memoria no inicializada, corre 4.715 forward passes durante veinte minutos, calcula 37.720 scores y devuelve un número.

Es el punto donde los dos defectos anteriores se combinan: **la descarga puede fallar en silencio (bug 1) y el parámetro sin inicializar convierte ese silencio en NaN o en colapso (bug 3).**

**El arreglo:**

```python
weights_file = 'torch_weights.h5'
assert os.path.isfile(weights_file), f'FALTA {weights_file} — revisa la celda 15'
assert os.path.getsize(weights_file) == 48474142, \
       f'{weights_file} truncado: {os.path.getsize(weights_file)} B, deberian ser 48474142'
print('loading weights from file [%s]' % weights_file)
r = network_eval.load_state_dict(torch.load(weights_file, map_location=device))
print(r)                                        # <All keys matched successfully>
network_eval = network_eval.to(device)
```

El segundo `assert` detecta una descarga truncada, que es el modo de fallo más probable. Y imprimir el retorno de `load_state_dict` convierte una carga exitosa en algo **visible**.

---

## 5. El `print` de 37.720 líneas

```python
for c, (p1, p2) in enumerate(zip(list1, list2)):
    ...
    print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))
```

**Dentro** del bucle, sin condición. Medido:

| | |
|---|---|
| líneas de output | **37.720** |
| volumen de texto | **1.321 KB = 1,3 MB** |

Escribir eso en memoria es gratis (1,53 s con y sin `print`). El problema es que en Colab cada línea viaja del kernel al navegador por websocket y el frontend intenta renderizar las 37.720. Y como el enunciado pide entregar el `.ipynb` **con todas las celdas ejecutadas**, quedan embebidas en el JSON.

**Lo que efectivamente pasó en esta ejecución:**

```
Se han truncado las últimas 5000 líneas del flujo de salida.
```

Colab descartó **32.721 de las 37.720 líneas** y dejó el notebook en **703 KB**, en su mayoría scores que nadie va a leer. El output que se entrega está incompleto de todos modos.

**El arreglo, tres caracteres:**

```python
    if c % 5000 == 0: print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))
```

> Curiosidad útil del truncamiento: las 4.999 líneas que sobrevivieron resultaron ser los **últimos** pares del archivo, y sus etiquetas coinciden al **100 %** con las últimas 4.999 de `voxceleb1_veri_test.txt`. Eso permitió reconstruir las estadísticas por clase a posteriori — pero solo sobre un subconjunto de hablantes, cuyo EER local (2,36 %) es más fácil que el global (3,19 %).

---

## 6. Ineficiencias que sí son inocuas (y por qué decirlo importa)

Vale distinguir un defecto de una ineficiencia irrelevante. Estas son las segundas, medidas:

**La búsqueda lineal.** `np.where(unique_list == p1)[0][0]` compara contra los 4.715 strings uno por uno, sobre un array que `np.unique` ya devolvió **ordenado**:

| Método | Tiempo para las 75.440 búsquedas |
|---|---|
| `np.where(...)` (el del lab) | **1,53 s** |
| `np.searchsorted(...)` (binaria) | 0,08 s |
| `dict` (tabla hash) | **0,012 s — 97× más rápido** |

Un `dict` es 97× más rápido… y ahorra 1,5 segundos. **No vale la pena tocarlo.** Es un patrón que importaría con 100.000 audios y 10 M de pares (esos 1,5 s serían horas), no acá.

**El broadcast de 5 dimensiones de `VladPooling`.** Materializa `(1, 1, T/16, 10, 512)` cuando la forma factorizada `(Aᵀ X) − (Σa)c` no lo necesita. Para el audio típico son 2 MB y décimas de milisegundo; con 4.000 descriptores serían 78 MB y 32× más lento. Irrelevante en inferencia con `batch=1`, insostenible al escalar.

**El `labels` redundante.** `labels += [verify_lb[c]]` reconstruye elemento por elemento un array que ya existe: al final `labels` es idéntico a `verify_lb`.

**Las dos transposiciones que se cancelan.** `lin_spectogram_from_wav` devuelve `linear.T` y `load_data` hace `mag.T` acto seguido. Se podrían borrar ambas.

**`np.stack` en lugar de `np.concatenate`.** Produce `(4715, 1, 512)` con un eje del medio que es el `batch_size=1` fosilizado, y obliga a la indexación `feats[ind1, 0]` que en la celda 27 parece arbitraria.

**`np.loadtxt(f, str)`.** Produce `dtype='<U29'`: NumPy mide la cadena más larga y **rellena todas** a 29 caracteres × 4 bytes. Resultado: **12,5 MB en RAM para un archivo de texto de 2,2 MB** (factor 5,7×), y la columna de etiquetas ocupa 116 bytes por celda para almacenar un bit. Irrelevante a esta escala; trampa de memoria en cuanto los datos crecen.

---

## 7. Dos trampas de ejecución

**El orden de las celdas.** La clase `NetVlad` (celda 12) referencia `VladPooling`, que se define en la **celda 13**, después. En Python funciona porque el nombre se resuelve al *instanciar* (celda 21), no al *definir* la clase. Pero ejecutar la celda 12 y saltar directo a la 21 da `NameError: name 'VladPooling' is not defined`.

**El `.eval()` que no está donde parece.** La celda 21 carga los pesos pero **no** llama a `network_eval.eval()`; eso ocurre en la celda 22. En el flujo normal está bien, pero cualquier forward de prueba entre ambas correría con las **38 BatchNorms en modo train**, recalculando estadísticas del batch y **sobreescribiendo los `running_var`** recién cargados. Con entradas normalizadas por frame y `batch_size=1`, esas estadísticas no se parecen a las de VoxCeleb2, y los embeddings pasarían a depender de **cuántos audios se procesaron antes**.

Y es peor de lo que suena por el hallazgo de [El checkpoint abierto](04-el-checkpoint-abierto): **3.518 canales tienen `running_var = 0` con filtros encogidos a 10⁻³³**. En modo `eval` emiten su `beta` constante y son inofensivos. En modo `train` dejarían de hacerlo y empezarían a amplificar ruido de 10⁻³³ dividido por la desviación del batch.

> `torch.no_grad()` **no** sustituye a `.eval()`: desactiva el registro de gradientes, no el modo train de BatchNorm. Son dos cosas distintas y ambas hacen falta.

---

## 8. Resumen

| # | Defecto | Severidad | Arreglo |
|---|---|---|---|
| 1 | `&` sin comillas → `wget` en segundo plano | **alta** — hace fallar el `unzip` | entrecomillar la URL |
| 2 | `unzip -q` sin `-n`/`-o` | **alta** — cuelga la celda | añadir `-n` y una guardia |
| 3 | `nn.Parameter(torch.Tensor(...))` sin inicializar | **alta** — NaN no reproducible | `torch.randn(...) * 0.01` |
| 4 | `if os.path.isfile` sin `else` | **alta** — fallo silencioso | `assert` con tamaño esperado |
| 5 | `print` de 37.720 líneas | media — output truncado, 703 KB | `if c % 5000 == 0` |
| 6 | `spec_len`, `freq/time`, `sys` sin usar | baja — residuos del port | borrar |
| 7 | búsqueda lineal, broadcast 5-D, `np.stack` | **nula acá** | no tocar |

Los cuatro primeros comparten una propiedad: **fallan sin decirlo.** Ninguno lanza una excepción en el momento en que ocurre el problema; los cuatro producen su síntoma más tarde, en otra celda, disfrazado de otra cosa. Es la categoría de defecto que más tiempo cuesta, y la que un `assert` bien puesto elimina.

---

**Anterior:** [El EER, el umbral y la dirección común](05-el-eer-y-la-direccion-comun) · **Siguiente:** [Las tres actividades](07-las-tres-actividades)
