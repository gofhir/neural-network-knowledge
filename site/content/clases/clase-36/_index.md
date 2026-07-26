---
title: "Clase 36 - Introducción al Análisis de Video"
weight: 360
sidebar:
  open: true
---

**Profesor:** Vladimir Araujo (Senior AI Researcher)
**Módulo:** Audio y Video (parte de video)

Segunda parte del módulo de Audio y Video, ahora sobre **video**. La visión por computador maduró entendiendo **imágenes**; el video —una secuencia de imágenes— recibió menos atención pese a ser el formato dominante del mundo real. La clase introduce el campo: qué es un video y por qué el **movimiento** lo cambia todo, las dos grandes áreas (**seguimiento de objetos** y **reconocimiento de acciones**), los **datasets** que marcaron el progreso (de KTH y UCF101 a Kinetics y EPIC-KITCHENS), y la evolución de los **enfoques de deep learning** —del 2D CNN por frame, que ignora el orden temporal, a las arquitecturas que sí lo modelan (2D CNN + RNN, two-stream con flujo óptico, convoluciones 3D).

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: definición, VOT y flujo óptico, action recognition, datasets, enfoques de deep learning" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: restricción del flujo óptico, invarianza al orden del 2D CNN, LRCN (CNN+LSTM), two-stream, convolución 3D, inflado I3D, consenso TSN" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Clasificación de video con 2D CNN + fusión temporal, y modelado del tiempo (CNN+LSTM / Conv3D) en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-36" title="Laboratorio: Video Understanding" subtitle="Clasificación de acciones sobre UCF11 con backbone ResNet y muestreo de frames" icon="variable" >}}
  {{< card link="/clases/clase-37" title="Clase siguiente: Datasets y Herramientas para Audio" subtitle="El ciclo de vida del dato de audio: formatos, transforms, augmentation, datasets" icon="arrow-right" >}}
  {{< card link="/clases/clase-35" title="Clase anterior: Análisis de Audio" subtitle="Fourier, FFT, sampling, STFT, MFCC" icon="arrow-left" >}}
  {{< card link="/clases/clase-11" title="Relacionada: Redes Recurrentes" subtitle="RNN/LSTM — la base del 2D CNN + RNN para video" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de Video" subtitle="Video, movimiento, stream vs sequence, VOT y action recognition" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de Acciones" subtitle="Tareas, datasets, y la evolución de enfoques de deep learning" icon="book-open" >}}
  {{< card link="/fundamentos/flujo-optico" title="Flujo óptico" subtitle="Desplazamiento de píxeles, correspondencia, Horn-Schunck, FlowNet" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Datasets

{{< cards >}}
  {{< card link="/papers/ucf101-soomro-2012" title="UCF101 (2012)" subtitle="Soomro et al. — 101 acciones de YouTube 'in the wild'" icon="document-text" >}}
  {{< card link="/papers/hmdb-kuehne-2011" title="HMDB (2011)" subtitle="Kuehne et al. — 51 acciones de películas y web" icon="document-text" >}}
  {{< card link="/papers/kinetics-kay-2017" title="Kinetics (2017)" subtitle="Kay et al. — el 'ImageNet del video'" icon="document-text" >}}
  {{< card link="/papers/something-something-goyal-2017" title="Something-Something (2017)" subtitle="Goyal et al. — interacciones que exigen razonamiento temporal" icon="document-text" >}}
  {{< card link="/papers/epic-kitchens-damen-2018" title="EPIC-KITCHENS (2018)" subtitle="Damen et al. — visión egocéntrica, untrimmed" icon="document-text" >}}
{{< /cards >}}

### Métodos

{{< cards >}}
  {{< card link="/papers/flownet-dosovitskiy-2015" title="FlowNet (2015)" subtitle="Dosovitskiy et al. — flujo óptico con CNN" icon="document-text" >}}
  {{< card link="/papers/two-stream-simonyan-2014" title="Two-Stream (2014)" subtitle="Simonyan & Zisserman — apariencia + movimiento (flujo óptico)" icon="document-text" >}}
  {{< card link="/papers/lrcn-donahue-2015" title="LRCN (2015)" subtitle="Donahue et al. — CNN + LSTM (el '2D CNN + RNN' de la clase)" icon="document-text" >}}
  {{< card link="/papers/c3d-tran-2015" title="C3D (2015)" subtitle="Tran et al. — convoluciones 3D espacio-temporales" icon="document-text" >}}
  {{< card link="/papers/i3d-carreira-2017" title="I3D (2017)" subtitle="Carreira & Zisserman — inflar 2D a 3D; pre-entrenar en Kinetics" icon="document-text" >}}
  {{< card link="/papers/tsn-wang-2016" title="TSN (2016)" subtitle="Wang et al. — muestreo esparcido por segmentos (la idea del lab)" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/video" title="Dominio: Video" subtitle="Línea de tiempo: de las CNN 3D y two-stream a los foundation models de video (Sora, Veo)" icon="globe-alt" >}}
{{< /cards >}}
