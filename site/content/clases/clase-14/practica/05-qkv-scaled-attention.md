---
title: "05 - Q/K/V con scaling: el primer ladrillo del Transformer"
weight: 50
math: true
---

En los capitulos anteriores construimos las piezas: embeddings, dot product, cross-entropy, autograd, un mini-Word2Vec. Ahora viene el salto. Este capitulo es donde el Transformer empieza a existir de verdad. Vamos a tomar la atencion "degenerada" del primer escalon — esa que era casi una identidad disfrazada — y la vamos a convertir en self-attention real, con tres proyecciones aprendibles y el famoso scaling por $\sqrt{d_k}$.

El script que acompana este capitulo es `clase_14/practica/02_qkv_scaled_attention.py`. Te recomiendo leer hasta el final de la seccion 4, correr el script, mirar los numeros, y volver al texto para las secciones que faltan.

---

## 1. Recap: las dos limitaciones del escalon 1

En el escalon 1 hicimos self-attention "desnuda": $Q = K = V = X$. Es decir, cada palabra se usaba a si misma como query, como key y como value. Tomamos los embeddings, los multiplicamos por su transpuesta, aplicamos softmax, y la salida era una suma ponderada de los mismos embeddings.

Funciono mecanicamente: la matematica corrio sin errores, las shapes calzaron, el softmax sumo a 1. Pero **el modelo no estaba transformando nada util**. Mira la matriz de pesos que produjo:

```
weights ESCALON 1 (Q=K=V=X, sin scaling):
[1.000, 0.000, 0.000, 0.000]   <- 'I' atiende solo a si misma
[0.008, 0.991, 0.000, 0.001]   <- 'love' atiende solo a si misma
[0.000, 0.000, 0.991, 0.009]   <- 'neural' atiende solo a si misma
[0.003, 0.002, 0.063, 0.932]   <- 'networks' atiende casi solo a si misma
```

Cada fila tiene un valor cercano a 1 en la diagonal y casi cero en el resto. Eso significa que la "atencion" colapso a una operacion trivial: cada palabra se queda con su propio embedding. La salida $\text{output} \approx X$. Nada se mezcla. Ningun contexto se incorpora. El modelo no esta aprendiendo a contextualizar.

Las dos razones son:

- **$Q = K = V = X$ es rigido.** Una misma palabra esta cumpliendo tres roles distintos con el mismo vector. No hay flexibilidad: la palabra no puede "preguntar una cosa" y "presentarse como otra".
- **Sin scaling el softmax satura.** Los dot products entre embeddings random de dimension media-alta son numeros grandes. Al pasarlos por softmax, uno se lleva casi toda la masa y los demas quedan en cero. La distribucion se vuelve casi one-hot. Como el gradiente del softmax depende de la dispersion de la masa, una distribucion saturada tiene gradiente practicamente nulo. **El modelo no puede aprender** porque el error no propaga.

Este capitulo arregla ambas cosas. Vamos a meter tres matrices aprendibles distintas para Q, K y V — una para cada rol — y vamos a dividir los scores por $\sqrt{d_k}$ para domar al softmax.

{{< concept-alert type="recordar" >}}
La atencion "degenerada" del escalon 1 era una version pedagogica: sirvio para entender la mecanica (dot product → softmax → suma ponderada) sin parametros aprendibles que distrajeran. Pero ningun Transformer real funciona asi. El escalon 2, este, es donde el modelo gana capacidad de aprender.
{{< /concept-alert >}}

---

## 2. Que es una "proyeccion lineal"

Antes de meter Q, K, V hay que tener clara una operacion fundamental: **multiplicar un vector por una matriz**. Esa operacion tiene un nombre tecnico — proyeccion lineal — pero la idea es muy concreta. Multiplicar un vector por una matriz lo **transforma**. Le cambia la direccion, la magnitud, o ambas.

Lo mejor es verlo en 2D, donde podemos dibujar.

### 2.1 Identidad: no cambia nada

$$
v = \begin{bmatrix} 3 \\ 2 \end{bmatrix}, \quad I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad I \cdot v = \begin{bmatrix} 3 \\ 2 \end{bmatrix}
$$

La matriz identidad multiplicada por $v$ devuelve $v$. Es como multiplicar por 1 en escalares.

### 2.2 Escalar: estira o comprime componentes

$$
S = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad S \cdot v = \begin{bmatrix} 6 \\ 1 \end{bmatrix}
$$

La componente x se duplico, la componente y se redujo a la mitad. La matriz "decide" que tan importante es cada eje.

### 2.3 Rotar: cambia direccion

$$
R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \quad R \cdot v = \begin{bmatrix} -2 \\ 3 \end{bmatrix}
$$

Esa matriz es una rotacion de 90 grados. El vector $v=(3,2)$ se transforma en $(-2,3)$: misma longitud, distinta direccion.

### 2.4 Proyectar/achatar: pierde informacion

$$
P = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad P \cdot v = \begin{bmatrix} 3 \\ 0 \end{bmatrix}
$$

Esta matriz "tira" la componente y a cero. Es una proyeccion al eje x. La operacion perdio informacion: a partir del resultado no podemos recuperar el $v$ original. **Perder informacion no es necesariamente malo**: a veces el modelo necesita quedarse solo con un aspecto del input y descartar el resto.

### 2.5 La idea fuerza

En general, multiplicar $v \cdot W$ (o $W \cdot v$, segun convencion) toma un vector de entrada y devuelve **otro vector**, posiblemente en un espacio de distinta dimension. La matriz $W$ es la regla de transformacion.

$$
x \cdot W = \text{nuevo vector transformado}
$$

Una **proyeccion lineal** es exactamente esto: aplicar una matriz $W$ a un vector $x$ para obtener $xW$. La palabra "proyeccion" en el contexto del Transformer es un poco abusiva — no siempre achatamos a una dimension menor — pero el termino quedo. Para nuestros propositos, "proyeccion lineal" = "aplicar una matriz aprendible".

En PyTorch, esto se hace con `nn.Linear(in_dim, out_dim, bias=False)`. Esa capa **es** una matriz, con la diferencia de que esta registrada como parametro aprendible: autograd la trackea, el optimizer la actualiza.

```python
import torch
import torch.nn as nn

W = nn.Linear(4, 4, bias=False)   # matriz aprendible 4x4
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = W(x)                          # proyeccion: y = x @ W.weight.T
```

{{< concept-alert type="clave" >}}
Una proyeccion lineal es una matriz aplicada a un vector. La matriz transforma: estira, rota, achata. En el Transformer, **cada proyeccion es aprendible**: el modelo descubre que transformacion conviene aplicar para resolver la tarea.
{{< /concept-alert >}}

---

## 3. Por que tres matrices: Q, K, V como tres facetas del mismo embedding

Listo. Ya entendemos que multiplicar un embedding por una matriz lo transforma. Ahora la pregunta del millon: **por que necesitamos tres matrices distintas y no una sola?**

La respuesta corta es: porque cada palabra cumple tres roles distintos en el mecanismo de atencion, y necesita un vector distinto para cada rol.

### 3.1 Analogia del perfil de LinkedIn

Imagina que cada palabra es una persona con un perfil de LinkedIn lleno de informacion:

- nombre, edad, profesion actual
- estudios, certificaciones, idiomas
- intereses profesionales, hobbies
- empresas en las que trabajo

El embedding $X[i]$ es ese perfil completo: una lista de 8 (o 512, o 4096) numeros que codifican todo lo que sabemos de la palabra.

Cuando esa persona interactua con un buscador de talento, juega tres roles distintos segun el momento:

1. **Cuando BUSCA un colaborador**, lo que importa son sus intereses ("estoy buscando alguien que sepa de FHIR"). De su perfil completo, el sistema extrae solo los campos relevantes para "que esta buscando". Eso es el rol **query**.
2. **Cuando SE PRESENTA al sistema** para que la encuentren, lo que importa es su profesion y especialidad ("soy ingeniero de software con foco en interoperabilidad"). De su perfil completo, el sistema extrae solo los campos relevantes para "como aparezco en busquedas ajenas". Eso es el rol **key**.
3. **Cuando APORTA INFO** una vez que alguien la encontro, lo que importa es su conocimiento real ("aqui van 5 paginas de mi experiencia con HL7"). De su perfil completo, el sistema extrae el contenido sustantivo. Eso es el rol **value**.

Mismo perfil, tres facetas. Las tres facetas no son arbitrarias: cada una destila un aspecto distinto del perfil para un proposito distinto. **Eso es exactamente lo que hacen $W^Q$, $W^K$, $W^V$**: son tres funciones (matrices) que extraen tres facetas distintas del mismo embedding.

$$
q_i = x_i W^Q \quad k_i = x_i W^K \quad v_i = x_i W^V
$$

Cada token $i$ tiene un solo embedding $x_i$, pero produce tres vectores distintos: su query $q_i$, su key $k_i$, su value $v_i$.

### 3.2 Diagrama

```
              x  (embedding original)
                       |
        +--------------+--------------+
        |              |              |
      x @ W_Q        x @ W_K        x @ W_V
        |              |              |
        v              v              v
        q              k              v
    (query)         (key)         (value)
   "que           "como me       "que
    pregunto"      presento"      entrego"
```

Las tres matrices $W^Q$, $W^K$, $W^V$ son **independientes**: tienen sus propios pesos, sus propios gradientes, su propia historia de actualizaciones. Todas reciben el mismo input $x$, pero producen tres salidas distintas.

### 3.3 Por que separar identidad y contenido

Un detalle importante: el rol de **key** y el rol de **value** son distintos a proposito.

- $K$ es la **etiqueta** que la palabra usa para hacerse encontrar. Es comparable. Tiene que vivir en el mismo espacio que las queries.
- $V$ es el **contenido** que la palabra entrega cuando es seleccionada. No tiene por que ser comparable con nada: solo tiene que ser informativo.

En la analogia del LinkedIn: la *key* es como el resumen profesional que aparece en los resultados de busqueda; el *value* es el CV completo que se entrega cuando alguien hace click. Si pusieras el CV completo en los resultados de busqueda, el matching seria ineficiente y ruidoso. Si pusieras solo el resumen como contenido a entregar, el resultado seria pobre.

Volveremos a esto en la seccion 7. Por ahora, queda la idea: **3 matrices porque hay 3 roles distintos**.

---

## 4. Pregunta profunda: como saben las matrices su rol?

Aqui llegamos al primer momento de confusion clasica. Si las tres matrices $W^Q$, $W^K$, $W^V$ se inicializan random, son indistinguibles entre si, y las tres reciben el mismo input $x$... **como rayos saben cual es la query, cual la key y cual la value?**

La respuesta es contraintuitiva pero clarisima una vez que hace click:

> **Las matrices no saben nada. Son tres montones de numeros, identicos en estructura, distintos solo por sus valores random iniciales. El rol viene de COMO se usan en el codigo.**

### 4.1 Es la posicion en el grafo, no la identidad

Mira el forward pass de self-attention, anotado:

```python
def forward(self, x):
    Q = self.W_Q(x)                              # W_Q se usa AQUI
    K = self.W_K(x)                              # W_K se usa AQUI
    V = self.W_V(x)                              # W_V se usa AQUI

    scores = Q @ K.transpose(-2, -1)             # Q y K se multiplican
    scores = scores / math.sqrt(self.d_k)        # scaling
    weights = F.softmax(scores, dim=-1)          # normalizacion

    output = weights @ V                         # V se usa AQUI, despues del softmax

    return output, weights
```

Si intercambiaras `self.W_Q` y `self.W_K` en el codigo, el modelo seguiria funcionando. Lo que llamabas $W_Q$ pasaria a jugar el rol de key, y al reves. Las matrices no tienen un sticker que diga "yo soy la query": el rol viene de **donde aparece la matriz en el grafo de computo**.

### 4.2 Como las moldea backprop

Ahora viene la parte bonita. Cuando llamas `loss.backward()`, autograd recorre el grafo de computo al reves desde el loss hacia las hojas. Cada matriz aprendible tiene un **camino distinto** hacia el loss:

- $W^Q$: $x \to Q \to \text{scores} \to \text{softmax} \to \text{weights} \to \text{output} \to \text{loss}$
- $W^K$: $x \to K \to \text{scores} \to \text{softmax} \to \text{weights} \to \text{output} \to \text{loss}$
- $W^V$: $x \to V \to \text{output} \to \text{loss}$ (NO pasa por scores ni por softmax)

Mira la diferencia: $W^V$ tiene un camino mucho mas corto. No participa en el calculo de los pesos de atencion, solo en la mezcla final. $W^Q$ y $W^K$ pasan por la matriz de scores y por el softmax, lo que cambia completamente la forma del gradiente que reciben.

Como los caminos son distintos, **los gradientes que llegan a cada matriz son distintos**. Eso significa que las actualizaciones del optimizer son distintas. Despues de N pasos de entrenamiento, las tres matrices ya no son intercambiables: cada una se ha movido en una direccion distinta del espacio de parametros, guiada por gradientes distintos.

### 4.3 Analogia de los tres trabajadores en una fabrica

Imagina una fabrica que recibe tres trabajadores nuevos el mismo dia. Los tres tienen identica formacion, identica edad, identica capacidad. La gerencia los pone en tres puestos distintos: uno en recepcion de pedidos, otro en empaque, otro en despacho.

El primer dia los tres son intercambiables. Pero cada puesto tiene desafios distintos:

- En recepcion, llegan errores tipo "el cliente pidio mal". El trabajador aprende a verificar pedidos.
- En empaque, llegan errores tipo "se rompe la caja". El trabajador aprende a reforzar embalajes.
- En despacho, llegan errores tipo "se entrega tarde". El trabajador aprende a optimizar rutas.

Despues de 100 dias, cada trabajador tiene experticia distinta. Si los intercambiaras, la fabrica funcionaria peor. **Y nadie les enseno explicitamente "tu eres el de recepcion"**: la especializacion emergio de la asimetria del entorno.

Eso es exactamente lo que pasa con $W^Q$, $W^K$, $W^V$. Empiezan identicas en distribucion (random), pero como cada una recibe gradientes distintos por su posicion en el grafo, terminan especializandose en su rol.

{{< concept-alert type="clave" >}}
Los componentes de un modelo no tienen "roles intrinsecos". Sus roles emergen de **como se conectan en el grafo de computo**. Backprop ajusta cada componente segun su posicion en ese grafo. La especializacion es consecuencia de la asimetria estructural, no de instrucciones explicitas.
{{< /concept-alert >}}

### 4.4 Bonus: que pasa si las inicializas iguales

Pregunta natural: si lo unico que las distingue son los valores random iniciales, que pasa si las inicializo todas con los mismos numeros?

Respuesta: **se quedarian siempre iguales**. Si $W^Q = W^K = W^V$ al inicio, entonces en el primer forward pass $Q = K = V$, y los gradientes que llegan a las tres son iguales (modulo la asimetria del grafo, que en este caso especifico tampoco rompe la simetria entre Q y K). Como reciben el mismo gradiente, el optimizer las actualiza igual, y siguen siendo iguales.

Es el problema clasico de la **simetria no rota** en redes neuronales. Por eso PyTorch inicializa las capas con valores random distintos por defecto: para que el ruido inicial rompa la simetria y permita que cada matriz tome su propio camino. El random inicial es el "empujoncito" que permite que la asimetria del grafo haga su trabajo.

---

## 5. Por que la matriz de scores ya NO es simetrica

Otro momento de confusion clasica. En el escalon 1, la matriz de scores era $X X^\top$, que es simetrica por construccion: $\text{score}[i,j] = x_i \cdot x_j = x_j \cdot x_i = \text{score}[j,i]$ porque el producto punto es conmutativo.

Ahora la matriz de scores es $Q K^\top$. Es simetrica? **No.** Y eso es bueno.

### 5.1 La pregunta linguistica

Pensemoslo en lenguaje natural. Considera la oracion "perro mira a amor" y su inversa "amor mira a perro". Cuando dices "perro mira a amor", el rol de cada palabra es distinto: perro es el sujeto que observa, amor es el objeto que se observa. Si pides al modelo "como atiende perro a amor", la respuesta no tiene por que ser igual a "como atiende amor a perro". **La atencion es direccional**.

En el escalon 1, la matriz simetrica forzaba al modelo a tratar igual ambas direcciones: el peso que perro pone en amor era exactamente el peso que amor pone en perro. Eso es una restriccion artificial: el lenguaje no funciona asi.

### 5.2 Ejemplo numerico concreto

Hagamoslo con numeros pequenos. Supongamos $d_{model} = d_k = 2$ y dos tokens:

$$
X[0] = \begin{bmatrix} 3 \\ 2 \end{bmatrix} \quad (\text{"perro"}), \quad X[1] = \begin{bmatrix} 1 \\ 5 \end{bmatrix} \quad (\text{"ladra"})
$$

Las matrices (random pero fijas para el ejemplo):

$$
W^Q = \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

Calculamos las queries y las keys:

$$
Q[0] = X[0] \cdot W^Q = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad Q[1] = X[1] \cdot W^Q = \begin{bmatrix} 1 \\ -4 \end{bmatrix}
$$

$$
K[0] = X[0] \cdot W^K = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad K[1] = X[1] \cdot W^K = \begin{bmatrix} 5 \\ 1 \end{bmatrix}
$$

Ahora la matriz de scores. Veamos los dos elementos cruzados:

$$
\text{score}[0,1] = Q[0] \cdot K[1] = 3 \cdot 5 + 1 \cdot 1 = 16
$$

$$
\text{score}[1,0] = Q[1] \cdot K[0] = 1 \cdot 2 + (-4) \cdot 3 = -10
$$

**16 vs -10**. Numeros completamente distintos. La matriz de scores NO es simetrica.

Comparemos con lo que hubiera pasado en el escalon 1, donde $Q = K = X$:

$$
\text{score}_{\text{escalon 1}}[0,1] = X[0] \cdot X[1] = 3 \cdot 1 + 2 \cdot 5 = 13
$$

$$
\text{score}_{\text{escalon 1}}[1,0] = X[1] \cdot X[0] = 1 \cdot 3 + 5 \cdot 2 = 13
$$

Identicos. La conmutatividad del dot product garantiza la simetria cuando $Q = K$.

### 5.3 De donde viene la asimetria

Importante: la asimetria **no viene** de cambiar el dot product (sigue siendo conmutativo). Viene de que ahora estamos haciendo dot product entre vectores **distintos**:

$$
\text{score}[i,j] = Q[i] \cdot K[j]
$$

$$
\text{score}[j,i] = Q[j] \cdot K[i]
$$

Como $Q[i] \neq K[i]$ y $Q[j] \neq K[j]$ (porque $W^Q \neq W^K$), los dos lados involucran **cuatro vectores distintos** en total, dos por cada lado. No hay ninguna razon estructural para que los dos productos den lo mismo.

Visualmente, en 2D, X[0] vive en algun punto del plano, Q[0] en otro, K[0] en otro. Q[0] apunta al noreste, K[0] al norte, X[0] al noreste pero con otra magnitud. Las matrices $W^Q$ y $W^K$ son rotaciones/escalados distintos del mismo input.

```
           Y
           ^
       Q[1].
           |
   K[0].   |   . Q[0]
           |
   --------+---------> X
           |
           |   . K[1]
           |
       (X[0] y X[1] viven en otra parte)
```

Cada uno apunta a su propia direccion. El producto punto entre Q[0] y K[1] mide cuanto se alinean **esos dos** vectores especificamente. Eso no tiene por que coincidir con cuanto se alinean Q[1] y K[0], que viven en posiciones distintas.

{{< concept-alert type="clave" >}}
La atencion del Transformer es **direccional** porque la matriz de scores no es simetrica. La asimetria emerge de tener dos proyecciones distintas $W^Q$ y $W^K$. Eso permite que el modelo aprenda relaciones del tipo "el sujeto atiende al verbo de cierta manera; el verbo atiende al sujeto de otra".
{{< /concept-alert >}}

---

## 6. Por que softmax y por que dot product

Las dos operaciones centrales del scaled dot-product attention son:

1. Calcular scores con un **producto punto**.
2. Convertirlos en distribucion con **softmax**.

Ambas elecciones podrian parecer arbitrarias. No lo son. Cada una es la mejor opcion para su rol.

### 6.1 Por que dot product

El dot product es la operacion natural para medir **similaridad geometrica** entre vectores. Geometricamente:

$$
a \cdot b = |a| \cdot |b| \cdot \cos\theta
$$

donde $\theta$ es el angulo entre los dos vectores. Eso significa:

- Vectores **alineados** (mismo angulo, $\cos\theta = 1$): producto punto positivo y grande.
- Vectores **perpendiculares** ($\cos\theta = 0$): producto punto cero.
- Vectores **opuestos** ($\cos\theta = -1$): producto punto negativo y grande en magnitud.

Esto es perfecto para atencion: queremos que la query "atienda" mas a las keys que apuntan en direccion similar, menos a las perpendiculares, y "rechace" (peso bajo) las opuestas.

Otras razones por las que el dot product gana:

- **Es barato**: $O(d)$ operaciones para vectores de dimension $d$.
- **Es diferenciable**: el gradiente es trivialmente calculable. $\frac{\partial (a \cdot b)}{\partial a} = b$.
- **Es paralelizable**: la matriz $Q K^\top$ se calcula con un solo `matmul` que las GPUs aceleran a fondo.

Alternativas que se descartaron:

- **Distancia euclidiana** $\|a - b\|^2$: tambien mide cercania, pero da numeros positivos y al hacer softmax sobre negativos no funciona igual (habria que negar). Ademas es mas costosa y menos natural para producir scores no acotados.
- **Cosine similarity** $\frac{a \cdot b}{|a| |b|}$: normaliza por magnitud, lo que tiene sentido en algunos contextos pero le quita una pieza de informacion al modelo. Ademas, agrega una division por componente que no es trivialmente paralelizable.
- **MLP que toma $(a, b)$ y devuelve un score**: mas expresivo, pero costoso. Esto es lo que hacia la "additive attention" de Bahdanau et al. (2014). Vaswani et al. eligieron dot product por velocidad.

### 6.2 Por que softmax

El softmax convierte un vector de numeros arbitrarios en una distribucion de probabilidad valida:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Propiedades clave:

- **Salidas positivas**: $e^{z_i} > 0$ siempre.
- **Suman 1**: por construccion $\sum_i \text{softmax}(z)_i = 1$.
- **Diferenciable**: el gradiente es bien conocido y suave.
- **Suave**: pequenos cambios en $z$ producen pequenos cambios en la salida (a diferencia de argmax).
- **Preserva orden**: el indice con score mas alto recibe el peso mas alto.

Alternativas que se descartaron:

- **Normalizar dividiendo por la suma** ($z_i / \sum_j z_j$): no funciona si hay scores negativos. Y no es robusto a outliers.
- **Sigmoid por componente**: no garantiza que sumen 1.
- **Argmax**: produce one-hot. **No es diferenciable**. Backprop muere ahi.
- **1 - p**: ni siquiera es una funcion de normalizacion bien definida.

### 6.3 La combinacion es la "interfaz" entre dos espacios

| Operacion | Que hace | Por que es la mejor |
|-----------|----------|---------------------|
| Dot product | Mide similaridad geometrica | Barato, diferenciable, paralelizable, captura $\cos\theta$ |
| Softmax | Convierte scores en distribucion | Salidas positivas, suma 1, suave, diferenciable |

{{< concept-alert type="clave" >}}
La combinacion **dot product + softmax** es la "interfaz natural" entre el espacio de embeddings (donde vives geometricamente) y el espacio de pesos (donde haces sumas ponderadas). Cada pieza es la mejor en su rol: dot product mide alineacion, softmax la convierte en mezcla valida.
{{< /concept-alert >}}

---

## 7. La estructura de la formula: la analogia del buscador

Ya entendemos por que tres matrices y por que dot product + softmax. Falta la pregunta mas estructural: **por que la formula tiene exactamente esta forma?**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Por que $V$ esta multiplicando despues del softmax y no antes? Por que $K$ y no $V$ se compara con $Q$? Por que necesitamos tres tensores y no dos?

La mejor analogia para entenderlo es **un buscador tipo Google**.

### 7.1 El buscador como mecanismo de atencion

Cuando haces una busqueda, hay tres "objetos" en juego:

1. **Tu query** ("recetas de pasta"). Es una expresion corta y especifica de lo que buscas.
2. **Los titulos de las paginas** indexadas. Son etiquetas cortas y densas, optimizadas para hacer matching.
3. **El contenido HTML completo** de cada pagina. Eso es lo que realmente quieres consumir cuando encuentras una buena pagina.

El proceso es:

```
Paso 1: tu query (Q) se compara contra los titulos (K)
        -> scores de relevancia

Paso 2: los scores se ordenan/normalizan (softmax)
        -> distribucion sobre paginas

Paso 3: se entrega el contenido (V) ponderado por relevancia
        -> resultado final
```

Mapeado al Transformer:

- **Query** $\equiv Q$: lo que la palabra "esta buscando" del contexto.
- **Key** $\equiv K$: la "etiqueta" que cada palabra exhibe para hacerse encontrar.
- **Value** $\equiv V$: el contenido sustantivo que cada palabra entrega cuando es seleccionada.

### 7.2 Por que K y V son distintos

Las **keys** son etiquetas pequenas, densas en informacion de "topic". Su unica funcion es matching: tienen que vivir en el mismo espacio que las queries y dar un buen score cuando hay relevancia.

Los **values** son contenido grande. No tienen que ser comparables: solo tienen que ser informativos. Pueden vivir en otro espacio. De hecho, en multi-head attention $V$ tiene una dimension distinta de $K$ a veces.

Si fusionaramos $K$ y $V$ en uno solo (volviendo a 2 vectores en lugar de 3), tendrias que elegir: o el mismo vector se usa para matching y para entrega, lo cual es restrictivo (forzas a que el "titulo" y el "contenido" sean lo mismo); o renuncias a una de las dos cosas.

Vaswani et al. eligieron mantenerlos separados. Tres vectores por token, tres roles.

### 7.3 Por que V esta despues del softmax

Aqui esta el detalle mas sutil de la formula. Mira de nuevo:

$$
\text{output} = \underbrace{\text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)}_{\text{pesos } \alpha_{ij}} \cdot V
$$

El softmax decide los **pesos**: cuanto de cada token se va a mezclar. Esos pesos viven en $[0,1]$ y suman 1. Despues, esos pesos multiplican $V$, que es el contenido a mezclar.

**Antes del softmax**: estamos en la fase de "decidir QUIEN es relevante". Comparas queries contra keys.

**Despues del softmax**: estamos en la fase de "transportar la INFORMACION del que fue elegido". Los values pasan por la mezcla ponderada.

Ahora pensemos en las alternativas:

- Si pusieras $V$ en el matching (es decir, si calcularas scores como $Q V^\top$): estarias comparando queries contra contenidos completos. Eso es ruidoso e impreciso. La query "recetas de pasta" no tiene por que alinearse con el HTML completo de una pagina; tiene que alinearse con su titulo.
- Si pusieras $K$ despues del softmax (es decir, si la salida fuera $\text{softmax}(...)\cdot K$): estarias mezclando etiquetas, no contenidos. La salida no tendria informacion sustantiva, solo etiquetas mezcladas. Inservible.

La estructura exacta es:

```
Q (lo que buscas) ─┐
                   ├─> dot product ─> softmax ─> pesos ─┐
K (etiquetas)    ─┘                                    │
                                                       ├─> output
                                            V (contenido) ─┘
```

Necesitas los tres vectores para tener simultaneamente:

- **Asimetria direccional** ($Q \neq K$): la matriz de scores no es simetrica, la atencion es direccional.
- **Separacion identidad/contenido** ($K \neq V$): la cosa con la que te buscan no tiene que ser la cosa que entregas.

Con dos vectores no se puede tener ambas. Con tres si.

{{< concept-alert type="recordar" >}}
$Q$ y $K$ van **antes** del softmax porque su rol es decidir relevancia. $V$ va **despues** del softmax porque su rol es entregar informacion una vez decidida la relevancia. Cambiar el orden rompe la semantica.
{{< /concept-alert >}}

---

## 8. La saturacion del softmax: por que $\sqrt{d_k}$

Llegamos al segundo gran arreglo del escalon. Tenemos las tres matrices, tenemos la formula con la estructura correcta. Pero falta una pieza: **el factor $1/\sqrt{d_k}$** que aparece dividiendo en la formula. De donde sale?

### 8.1 La teoria

Supon que $q$ y $k$ son dos vectores de dimension $d_k$, con componentes iid extraidas de una distribucion con media 0 y varianza 1. Entonces:

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

Cada termino $q_i k_i$ es producto de dos variables iid de varianza 1, asi que tiene media 0 y varianza 1. La suma de $d_k$ terminos iid de varianza 1 tiene varianza $d_k$. Por tanto:

$$
\text{Var}(q \cdot k) = d_k \quad \Rightarrow \quad \text{std}(q \cdot k) = \sqrt{d_k}
$$

Eso significa que **a medida que crece $d_k$, los scores crecen tipicamente como $\sqrt{d_k}$**. Para $d_k = 64$, los scores tienen std ~8. Para $d_k = 512$, std ~22.6.

### 8.2 El problema visceral

Que pasa cuando metes scores grandes en un softmax? Recordemos:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Si los $z$ tienen rango grande (digamos algunos valen 20, otros -15), el termino $e^{z_i}$ con el $z_i$ mas grande domina por completo. Los demas se vuelven despreciables. La distribucion colapsa a casi one-hot: un peso ~ 1, los demas ~ 0.

Eso es **softmax saturado**. Tres consecuencias malas:

1. La atencion se vuelve "dura": cada token atiende solo a uno, no a una mezcla.
2. La salida es discontinua respecto a pequenos cambios en la entrada. Una pequena perturbacion en $z$ puede cambiar el ganador.
3. **El gradiente del softmax se va a cero**: cuando una distribucion es casi one-hot, los gradientes son minusculos. Backprop deja de transmitir senal.

Resultado: el modelo no puede aprender. Sin scaling, los Transformers con $d_k > 64$ son intrenables.

### 8.3 La solucion: dividir por $\sqrt{d_k}$

Si dividimos los scores por $\sqrt{d_k}$ antes del softmax:

$$
\text{score}_{\text{scaled}} = \frac{q \cdot k}{\sqrt{d_k}}
$$

la varianza queda renormalizada a $\sim 1$ independientemente de $d_k$:

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = 1
$$

Ahora los scores tienen std ~1 sin importar la dimension. El softmax recibe numeros del orden de 1-3, no del orden de 20. La distribucion se mantiene suave, los gradientes fluyen, el modelo aprende.

### 8.4 El experimento numerico

El script ejecuta este experimento para $d_k \in \{4, 64, 512\}$. Los resultados, copiados directamente de la salida:

```
d_k = 4:
  std de scores SIN scaling = 1.70  (teoria sqrt(4) = 2.00)
  std de scores CON scaling = 0.85

d_k = 64:
  std de scores SIN scaling = 8.58  (teoria sqrt(64) = 8.00)
  std de scores CON scaling = 1.07

d_k = 512:
  std de scores SIN scaling = 31.3  (teoria sqrt(512) = 22.6)
  std de scores CON scaling = 1.39
```

La teoria calza. Ahora el efecto en el softmax — comparando la primera fila de pesos sin/con scaling:

```
                   SIN scaling                              CON scaling
d_k = 4    [0.107, 0.078, 0.013, 0.012, 0.759, 0.031]   [0.174, 0.149, 0.061, 0.059, 0.464, 0.094]
d_k = 64   [0.000, 0.942, 0.000, 0.000, 0.058, 0.000]   [0.063, 0.437, 0.111, 0.040, 0.309, 0.041]
d_k = 512  [0.000, 1.000, 0.000, 0.000, 0.000, 0.000]   [0.070, 0.727, 0.032, 0.084, 0.049, 0.038]
```

Mira la columna de la izquierda. En $d_k = 4$ ya hay un valor que se lleva 0.76 (mucho). En $d_k = 64$ uno se lleva 0.94 (casi todo). En $d_k = 512$, uno se lleva 1.00 (literalmente todo, los demas son cero a 3 decimales). **Eso es softmax muerto.**

Ahora mira la columna de la derecha. Con scaling, las distribuciones se mantienen razonables: el maximo nunca sube de 0.73, y siempre hay masa repartida. El modelo puede mezclar tokens, no esta forzado a elegir uno solo.

{{< concept-alert type="clave" >}}
El factor $1/\sqrt{d_k}$ es lo que hace al Transformer **entrenable** a escalas modernas. Sin el, modelos con $d_k > 64$ no aprenderian. Es una correccion de varianza simple, derivada de un argumento estadistico de tres lineas, pero su impacto practico es enorme.
{{< /concept-alert >}}

---

## 9. La implementacion completa: SelfAttention como nn.Module

Vamos a escribir la self-attention real, version idiomatic de PyTorch. Aca esta el codigo completo (la version del script, ligeramente compacta para pegar en el notebook):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention scaled dot-product de una sola cabeza.

    Input:  x de shape (batch, T, d_model)
    Output: (batch, T, d_v), pesos (batch, T, T)
    """
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.d_k = d_k

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)                              # (batch, T, d_k)
        K = self.W_K(x)                              # (batch, T, d_k)
        V = self.W_V(x)                              # (batch, T, d_v)

        scores = Q @ K.transpose(-2, -1)             # (batch, T, T)
        scores = scores / math.sqrt(self.d_k)        # scaling
        weights = F.softmax(scores, dim=-1)          # (batch, T, T)
        output = weights @ V                         # (batch, T, d_v)

        return output, weights
```

Ahora linea por linea.

### 9.1 `class SelfAttention(nn.Module)`

`nn.Module` es la **clase base de PyTorch** para definir cualquier capa o modelo. Te da:

- Registro automatico de parametros aprendibles (las `nn.Linear` que pongas como atributos).
- Soporte para `.to(device)`, `.train()`, `.eval()`.
- Compatibilidad con autograd y con el optimizer.

La forma idiomatic es: subclase + `__init__` (declara componentes) + `forward` (define el calculo).

### 9.2 `super().__init__()`

Llama al constructor de `nn.Module` para inicializar los hooks internos. Si te lo olvidas, PyTorch tira un error la primera vez que tratas de registrar un parametro.

### 9.3 `d_k = d_k or d_model`

Pequeno truco de Python: si `d_k` es `None`, usa `d_model`. Sirve para que la API sea ergonomica: si solo pasas `d_model`, se asume que `d_k = d_v = d_model`. Esto es lo tipico en single-head attention.

### 9.4 `nn.Linear(d_model, d_k, bias=False)`

Esta es **la matriz aprendible**. La firma es `nn.Linear(in_features, out_features, bias=True)`. Internamente, almacena un tensor `weight` de shape `(out_features, in_features)` y, si `bias=True`, un tensor `bias` de shape `(out_features,)`.

Cuando llamas `W_Q(x)`, PyTorch hace `x @ self.W_Q.weight.T` (mas el bias si existe). En self-attention el bias se suele desactivar — la teoria del paper original lo omite y empiricamente no aporta mucho.

Las tres `nn.Linear` registran sus pesos como parametros del modulo. Si despues haces `for p in attention.parameters()`, las tres aparecen.

### 9.5 `Q = self.W_Q(x)`

Aplica la proyeccion. El shape pasa de `(batch, T, d_model)` a `(batch, T, d_k)`. Igual para `K` y `V`.

### 9.6 `Q @ K.transpose(-2, -1)`

Aqui esta el producto matricial que produce los scores. `K.transpose(-2, -1)` intercambia las dos ultimas dimensiones de `K`: de `(batch, T, d_k)` a `(batch, d_k, T)`. Luego `Q @ K.T` da `(batch, T, T)`.

Por que `transpose(-2, -1)` y no `K.T`? Porque `K.T` solo funciona en tensores 2D. `transpose(-2, -1)` funciona en cualquier numero de dimensiones — soporta el batch dim al frente. Es la version segura para multi-batch.

### 9.7 `scores / math.sqrt(self.d_k)`

El scaling. Notese: `math.sqrt` (CPU, escalar Python) y no `torch.sqrt` (tensor en GPU). Aqui no hace diferencia porque es un escalar fijo, pero `math.sqrt` es mas eficiente para constantes que no requieren gradiente.

### 9.8 `F.softmax(scores, dim=-1)`

Softmax sobre la **ultima dimension**. Esa es la dimension de las "keys" en cada fila. Recordatorio: para cada query (fila $i$), normalizamos sobre todas las keys (columnas $j$), de modo que cada fila suma 1.

Si por error pones `dim=-2`, normalizarias sobre las queries para cada key. Eso seria distinto, sin sentido en este contexto, y silenciosamente roto. Es un bug clasico.

### 9.9 `weights @ V`

El paso final: cada fila de `weights` es una distribucion sobre tokens. Multiplicandola contra `V` se obtiene una **suma ponderada** de los values. Shape: `(batch, T, T) @ (batch, T, d_v) = (batch, T, d_v)`.

### 9.10 Llamarlo

```python
torch.manual_seed(42)
d_model = 8
T = 4
X = torch.randn(T, d_model)

attention = SelfAttention(d_model=d_model)
X_batched = X.unsqueeze(0)               # (1, T, d_model)
output, weights = attention(X_batched)   # (1, T, d_v), (1, T, T)
```

Notese el `unsqueeze(0)`: agregamos una dimension de batch al frente. Aunque sea un solo ejemplo, PyTorch espera batch siempre. Esa convencion (`(batch, seq, dim)`) la veras en todos los modelos modernos.

---

## 10. Comparacion lado a lado: escalon 1 vs escalon 2

Aqui los resultados reales que produce el script con la oracion `["I", "love", "neural", "networks"]` y `d_model = 8`. Mismos embeddings, distinta atencion.

**Weights ESCALON 1 (Q = K = V = X, sin scaling):**

```
[1.000, 0.000, 0.000, 0.000]   <- 'I' atiende solo a si misma
[0.008, 0.991, 0.000, 0.001]   <- 'love' atiende solo a si misma
[0.000, 0.000, 0.991, 0.009]   <- 'neural' tambien
[0.003, 0.002, 0.063, 0.932]   <- 'networks' casi solo a si misma
```

Cada fila tiene un valor cercano a 1 y los demas cerca de 0. El modelo no esta mezclando informacion entre tokens.

**Weights ESCALON 2 (Q, K, V aprendibles, con scaling):**

```
[0.271, 0.306, 0.215, 0.208]
[0.272, 0.246, 0.099, 0.383]
[0.431, 0.153, 0.293, 0.123]
[0.301, 0.170, 0.367, 0.162]
```

Distribucion suave. Cada token reparte atencion entre todos los demas. Esto **no** significa que el modelo este atendiendo "bien" — todavia no se entreno, los pesos son random — pero significa que **puede atender**. La estructura no esta colapsada. Cuando metas backprop con un objetivo real, el modelo va a poder mover esos pesos hacia donde sea util.

La diferencia es estructural:

| | Escalon 1 | Escalon 2 |
|---|-----------|-----------|
| Q, K, V | $= X$ (compartido) | proyecciones aprendibles distintas |
| Parametros aprendibles | 0 | 3 matrices ($W^Q, W^K, W^V$) |
| Matriz de scores | simetrica | no simetrica (atencion direccional) |
| Scaling | no | $1/\sqrt{d_k}$ |
| Softmax | saturado para $d_k$ medio-alto | controlado para cualquier $d_k$ |
| Output | $\approx X$ (degenerado) | realmente transformado |

---

## 11. Verificar que los gradientes fluyen

Lo ultimo que falta es comprobar que las tres matrices son **realmente aprendibles**: que despues de un `loss.backward()`, las tres reciben gradiente, y que el optimizer podria actualizarlas.

El script hace este experimento. Define un target ficticio (igual al input, solo para tener algo contra que comparar), calcula MSE, y llama `backward`:

```python
target = X_batched.clone()
output, _ = attention(X_batched)

loss = F.mse_loss(output, target)
loss.backward()

print(attention.W_Q.weight.grad.norm().item())
print(attention.W_K.weight.grad.norm().item())
print(attention.W_V.weight.grad.norm().item())
```

Salida real:

```
W_Q.weight.grad.norm() = 0.2295
W_K.weight.grad.norm() = 0.3357
W_V.weight.grad.norm() = 0.9419
```

Tres puntos:

1. **Las tres son distintas de cero**: autograd traceo el grafo entero (`x -> Q/K/V -> scores -> softmax -> weights -> output -> loss`) y el error se propago hasta cada matriz. Si alguna fuera cero, habria un problema (un detach, un parametro desconectado, algo).
2. **Las tres son distintas en magnitud**: el camino de cada matriz al loss es distinto, asi que los gradientes que reciben son distintos. La de $W^V$ es la mas grande porque es la que multiplica directamente al output sin pasar por el softmax. Las de $W^Q$ y $W^K$ son mas chicas porque sus gradientes pasan por la derivada del softmax, que en distribuciones suaves tiene magnitud moderada.
3. **Optimizer.step() funcionaria**: si despues de esto llamaramos `optimizer.step()`, el optimizer aplicaria la regla de actualizacion (SGD, Adam, lo que sea) usando esos gradientes y movera los pesos en la direccion que reduce el loss.

Con esto cerramos: tenemos self-attention real, parametrizada, escalable, entrenable.

---

## 12. Pausa de verificacion

Antes de seguir al siguiente escalon (multi-head attention), asegurate de poder responder estas preguntas con tus propias palabras. Si alguna te queda dudosa, vuelve a la seccion correspondiente.

1. **Que es una proyeccion lineal?**
   Multiplicar un vector por una matriz para obtener otro vector posiblemente en otro espacio. La matriz transforma: estira, rota, achata.

2. **Por que necesitamos 3 matrices Q/K/V y no una sola?**
   Porque cada palabra cumple tres roles distintos: query (que pregunta), key (como se presenta), value (que entrega). Tener 3 vectores distintos del mismo embedding permite asimetria direccional ($Q \neq K$) y separacion identidad/contenido ($K \neq V$). Con 2 vectores no se logran ambas.

3. **Como saben las matrices su rol?**
   No saben. Las tres empiezan random e indistinguibles. El rol viene de **como se usan en el codigo**: $W^Q$ se aplica para producir la query, $W^K$ para la key, $W^V$ para el value. Backprop, al recorrer caminos distintos en el grafo de computo, les da gradientes distintos, y eso las moldea con el tiempo hacia roles especializados.

4. **Por que la matriz de scores ya no es simetrica?**
   Porque ahora $\text{score}[i,j] = Q[i] \cdot K[j]$ y $\text{score}[j,i] = Q[j] \cdot K[i]$ involucran cuatro vectores distintos en total. No hay razon estructural para que coincidan. La asimetria permite atencion direccional: "perro atendiendo a ladra" puede ser distinto de "ladra atendiendo a perro".

5. **Por que se divide por $\sqrt{d_k}$?**
   Porque la varianza del producto punto crece con $d_k$. Sin scaling, los scores se vuelven grandes en magnitud, el softmax satura a casi one-hot, y el gradiente se va a cero. Dividir por $\sqrt{d_k}$ renormaliza la varianza a ~1 sin importar la dimension.

6. **Por que V esta despues del softmax y Q/K antes?**
   Porque Q y K son para **decidir relevancia** (matching de queries contra etiquetas). V es la **informacion a entregar** una vez que la relevancia esta decidida. Si pusieras V en el matching, comparas queries contra contenidos completos (ruidoso). Si pusieras K despues del softmax, mezclarias etiquetas en lugar de contenido (inutil).

---

## 13. Lo que sigue: multi-head attention

Acabamos de construir una **cabeza** de atencion. Funciona. Pero tiene una limitacion: produce **una sola distribucion de pesos por token**.

Considera la palabra "kicked" en "Alexis kicked the ball". Esa palabra deberia poder atender simultaneamente a:

- El **sujeto** "Alexis" (quien hace la accion).
- La **accion** "kicked" (la propia palabra, en algunas tareas).
- El **objeto** "the ball" (sobre que recae la accion).

Con una sola cabeza, el modelo tiene que comprometer: si pone mucho peso en el sujeto, le queda poco para el objeto. La distribucion suma 1, y un solo numero por par no captura tipos de relacion.

**Multi-head attention** resuelve esto. La idea: ejecutar $h$ atenciones en paralelo, cada una con sus propias $W^Q, W^K, W^V$ distintas, cada una operando en un subespacio distinto del embedding. Cada cabeza puede especializarse en un tipo de relacion: una en sujeto-verbo, otra en verbo-objeto, otra en correferencias, etc. Al final se concatenan.

Hyperparametros tipicos del paper de Vaswani et al. (2017):

- $d_{model} = 512$
- $h = 8$ cabezas
- $d_k = d_v = d_{model} / h = 64$ por cabeza

Notese que $d_k$ se reduce. La idea es: cada cabeza opera en un subespacio mas chico, pero hay 8 de ellas en paralelo. El costo computacional total es similar a una sola cabeza con $d_k = 512$, pero la expresividad es mayor.

Eso lo construimos en el siguiente capitulo.

---

## Codigo y siguiente capitulo

Codigo completo: `clase_14/practica/02_qkv_scaled_attention.py`

Volver al [hub de practica](..).
