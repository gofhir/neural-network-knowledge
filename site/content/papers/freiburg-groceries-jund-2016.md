---
title: "Freiburg Groceries Dataset"
weight: 100
math: true
---

{{< paper-card
    title="The Freiburg Groceries Dataset"
    authors="Jund, Abdo, Eitel, Burgard"
    year="2016"
    venue="arXiv 2016"
    pdf="/papers/freiburg-groceries-jund-2016.pdf"
    arxiv="1611.05799" >}}
Dataset de clasificacion de productos de supermercado del grupo de robotica de la Universidad de Freiburg. **5000 imagenes** repartidas en **25 categorias** de articulos de abarrotes (leche, arroz, cafe, cereal, especias, etc.), capturadas con **camaras de telefonos moviles** en tiendas, hogares y oficinas reales, bajo iluminacion, fondos y angulos no controlados. Acompana un **baseline CNN** que clasifica la categoria del producto y un protocolo de evaluacion sobre escenas degradadas. Concebido originalmente para percepcion de objetos en robots de servicio, el dataset se reaprovecha en el Laboratorio 21 como banco de imagenes con texto real (marcas en aleman) para tareas de OCR.
{{< /paper-card >}}

---

## El problema / motivacion

Los robots de servicio que operan en hogares y tiendas necesitan reconocer objetos cotidianos de consumo: distinguir una caja de cereal de un paquete de cafe, una botella de leche de un tarro de mermelada. Hacia 2016, los datasets de objetos disponibles para entrenar redes convolucionales (ImageNet, PASCAL VOC, COCO) cubrian categorias genericas y escenas web heterogeneas, pero no la distribucion especifica de un robot moviendose por un supermercado o una despensa: productos empaquetados, vistos de cerca, en estanterias atestadas, con oclusiones parciales y bajo la iluminacion irregular de un local comercial.

El gap que ataca el paper es de **dominio**, no de algoritmo. La motivacion es disponer de un corpus etiquetado con las categorias de abarrotes que un robot de servicio realmente encontrara, capturado con el tipo de sensor barato y ruidoso (la camara de un telefono) que tales robots montan, en condiciones de adquisicion no controladas. El objetivo declarado es habilitar y comparar metodos de clasificacion de productos de supermercado en condiciones realistas, en lugar de en fotografias de catalogo limpias.

## El dataset

| Atributo | Valor |
| --- | --- |
| Imagenes | 5000 |
| Categorias | 25 |
| Dominio | Productos de supermercado / abarrotes |
| Sensor | Camaras de telefonos moviles |
| Condiciones | Tiendas, hogares y oficinas; iluminacion y fondos no controlados |
| Granularidad de etiqueta | Categoria del producto (image-level) |
| Origen | Grupo de robotica autonoma, Universidad de Freiburg |

Las **25 categorias** son tipos de articulos de abarrotes de consumo cotidiano: leche, arroz, cafe, cereal, te, especias, harina, azucar, jugo, agua, refrescos, pasta, conservas, dulces, entre otras. La etiqueta es a nivel de imagen: cada foto se asocia a una clase de producto, no a cajas delimitadoras ni mascaras, lo que lo posiciona como un dataset de **clasificacion** (no de deteccion ni de segmentacion).

### Recoleccion en condiciones reales

El rasgo distintivo del corpus es su procedencia *in-the-wild*. Las imagenes no se tomaron en un estudio fotografico ni se descargaron de catalogos online, sino que fueron **capturadas con telefonos moviles** por personas que recorrieron tiendas, hogares y oficinas. Esto introduce de forma deliberada la variabilidad que un robot real enfrenta:

- **Iluminacion no controlada**: luz fluorescente de pasillo, luz natural junto a ventanas, sombras de estanterias.
- **Fondos heterogeneos**: otros productos detras, estanterias, manos, mesas.
- **Angulos y distancias variables**: tomas frontales, oblicuas, de cerca y de lejos.
- **Oclusiones parciales**: productos a medio tapar por otros o por la propia estanteria.
- **Ruido del sensor**: las camaras de telefono de la epoca aportan compresion JPEG, desenfoque por movimiento y balance de blancos inconsistente.

Esa diversidad escenica es precisamente lo que vuelve al dataset util mas alla de su proposito original: las imagenes contienen los **envases reales de los productos**, con sus logos, marcas y texto impreso visibles, fotografiados en condiciones de OCR genuinamente dificiles.

## Metodo / baseline

El paper acompana el dataset con un **baseline basado en redes convolucionales** para la tarea de clasificar la categoria del producto a partir de la imagen. La red es una CNN estandar de la epoca (arquitectura tipo AlexNet/GoogLeNet preentrenada en ImageNet y luego ajustada, *fine-tuned*, sobre las 25 categorias de Freiburg Groceries), entrenada de forma supervisada con la etiqueta de categoria como objetivo.

El protocolo de evaluacion separa el corpus en particiones de entrenamiento y prueba y mide la **exactitud de clasificacion** (accuracy) sobre las 25 clases. La eleccion de transfer learning desde ImageNet es la receta canonica de 2016: la red preentrenada aporta representaciones visuales genericas y el fine-tuning las especializa al dominio de abarrotes, mitigando el tamano modesto del dataset (5000 imagenes es pequeno para entrenar una CNN profunda desde cero).

El baseline cumple un doble rol: (1) demuestra que la tarea es abordable con la maquinaria estandar de deep learning de la epoca, y (2) fija una **linea base reproducible** contra la cual trabajos posteriores pueden comparar, que es la funcion clasica de un paper de dataset.

## Resultados clave

El resultado central no es una arquitectura nueva sino el **dataset mismo y su baseline**. Las aportaciones verificables son:

1. **Un corpus etiquetado de dominio especifico**: 5000 imagenes de 25 categorias de abarrotes, capturadas en condiciones realistas con camaras moviles, liberado publicamente para la comunidad de robotica y vision.
2. **Una linea base CNN** que clasifica la categoria del producto y establece el nivel de desempeno de referencia bajo el protocolo de evaluacion propuesto.
3. **La evidencia de la dificultad del dominio**: al medir sobre escenas con iluminacion, oclusion y fondo no controlados, el dataset expone que clasificar productos *in-the-wild* es sustancialmente mas duro que sobre fotografias de catalogo, justificando la existencia del benchmark.

La contribucion metodologica, como en otros papers de dataset, esta en **identificar correctamente un gap de dominio y producir el groundtruth que permite estudiarlo**, no en una innovacion de modelo.

## Limitaciones

1. **Escala modesta**: 5000 imagenes en 25 clases es pequeno para los estandares de deep learning, lo que obliga a depender de transfer learning desde ImageNet y limita la capacidad de entrenar modelos grandes desde cero.
2. **Granularidad image-level**: la etiqueta es la categoria del producto, sin cajas delimitadoras ni mascaras. No sirve directamente para deteccion ni segmentacion sin anotacion adicional.
3. **Cobertura geografica y cultural acotada**: los productos provienen del mercado aleman, con marcas y texto en aleman, lo que sesga la distribucion de envases respecto a otros paises.
4. **Sin anotacion de texto**: el dataset no fue concebido para OCR; el texto de marcas y etiquetas esta presente en las imagenes pero **no anotado** (no hay transcripciones ni cajas de texto como groundtruth). Cualquier uso para reconocimiento de texto requiere anotar o evaluar cualitativamente.
5. **Ruido de adquisicion**: la variabilidad que lo hace realista tambien introduce imagenes muy degradadas (desenfoque, sobreexposicion) cuya etiqueta puede ser ambigua incluso para un humano.

## Conexion con el Laboratorio 21

En el [Laboratorio 21](/laboratorios/lab-21) el dataset Freiburg Groceries se **reaprovecha fuera de su proposito original**. El corpus fue creado para *clasificar* la categoria del producto; en el laboratorio no se usa la etiqueta de categoria, sino que se explotan las **imagenes como banco de envases reales con texto impreso** para una tarea de OCR / scene text recognition con [ABCNet](/papers/abcnet-liu-2020).

Tres conexiones explicitas con el material del laboratorio:

1. **De clasificacion a lectura de marcas**: el laboratorio descarta la tarea de clasificacion del paper y aplica un *spotter* de texto sobre las imagenes para **leer las marcas y palabras impresas en los envases** (la marca, el nombre del producto, descriptores). El dataset funciona aqui como una fuente de escenas con texto en condiciones reales, no como un benchmark de categorias.

2. **Texto en aleman y transferencia cross-idioma**: como los productos provienen del mercado aleman, el texto impreso esta en **aleman**. El walkthrough del laboratorio observa lecturas limpias de palabras como `milch` (leche), `reis` (arroz), `honig` (miel), `zucker` (azucar), `tomaten` (tomates) y `bohnen` (porotos). El hallazgo conceptual es que un modelo de scene text recognition entrenado mayoritariamente sobre texto en **ingles** logra leer texto en aleman sin reentrenamiento. La razon es que ambos idiomas comparten el **alfabeto latino**: el reconocedor opera a nivel de glifos y secuencias de caracteres latinos, de modo que el idioma de la palabra es en gran medida irrelevante mientras el script coincida. Esto constituye una demostracion de **transfer zero-shot cross-idioma** dentro del mismo sistema de escritura.

3. **La distribucion del dominio importa**: el laboratorio ilustra una leccion transversal de [scene text recognition](/fundamentos/scene-text-recognition) y de la [Clase 21](/clases/clase-21): un modelo de OCR generaliza a un idioma nuevo si comparte alfabeto, pero su exito depende de que la **distribucion empirica del texto** (tipografias de envase, contraste, curvatura del empaque, iluminacion) se parezca a la de su entrenamiento. Freiburg Groceries, por ser *in-the-wild* y de un idioma distinto al de entrenamiento, es un caso de prueba elegante de hasta donde llega esa generalizacion.

En sintesis, el laboratorio toma un dataset de robotica de 2016 pensado para clasificar abarrotes y lo convierte en un experimento de OCR multilingue, mostrando que la barrera real para el transfer de un reconocedor de texto no es el idioma sino el **alfabeto** y la distribucion visual del texto.

## Notas y enlaces

- **Fundamentos relacionados**:
  - [Scene Text Recognition](/fundamentos/scene-text-recognition/)
  - [Deteccion de objetos](/fundamentos/deteccion-de-objetos/)
- **Papers conectados**:
  - [ABCNet (Liu CVPR 2020)](/papers/abcnet-liu-2020/) — el spotter de texto que el laboratorio aplica sobre las imagenes de Freiburg Groceries.
  - [Total-Text (Ch'ng et al. 2017)](/papers/total-text-chng-2017/) — benchmark de texto curvo sobre el que ABCNet reporta sus numeros.
- **Clase**: [Clase 21 — Reconocimiento de texto en escenas](/clases/clase-21/)
- **Lab**: [Laboratorio 21 — scene text recognition aplicado](/laboratorios/lab-21/)
- **Repositorio oficial**: [`PhilJd/freiburg_groceries_dataset`](https://github.com/PhilJd/freiburg_groceries_dataset)
</content>
</invoke>
