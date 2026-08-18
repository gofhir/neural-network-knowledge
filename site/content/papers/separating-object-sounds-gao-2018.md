---
title: "Learning to Separate Object Sounds by Watching Unlabeled Video (2018)"
weight: 470
math: true
---

{{< paper-card
    title="Learning to Separate Object Sounds by Watching Unlabeled Video"
    authors="Ruohan Gao, Rogerio Feris, Kristen Grauman (UT Austin / IBM Research / Facebook AI Research)"
    year="2018"
    venue="ECCV 2018 / arXiv:1804.01665"
    arxiv="1804.01665"
    pdf="/papers/separating-object-sounds-gao-2018.pdf" >}}
Separación de fuentes sonoras guiada por los **objetos** que aparecen en el video, no por los hablantes. Aprende de video sin etiquetar qué objeto produce qué componente del sonido, combinando una red visual con **factorización de matrices no negativas** sobre el espectrograma. Es el complemento de [Looking to Listen](/papers/looking-to-listen-ephrat-2018) en la [Clase 44](/clases/clase-44): mientras aquel aísla voces usando rostros, este aísla instrumentos, motores o animales usando la detección de objetos.
{{< /paper-card >}}

---

## La idea

La NMF descompone un espectrograma en un diccionario de **bases espectrales** y sus activaciones temporales, todas no negativas. El problema clásico es que esas bases no tienen etiqueta: se sabe que hay $k$ componentes, no a qué corresponden.

El aporte del trabajo es usar el video para **asignar bases a objetos**. Se detectan los objetos presentes, y a lo largo de muchos videos sin etiquetar se aprende qué bases espectrales tienden a aparecer cuando aparece cada objeto. Es aprendizaje de correspondencia con instancias múltiples, sobre datos que nadie anotó.

Combina así una técnica clásica de procesamiento de señales, interpretable y con pocos parámetros, con una red neuronal moderna para la parte visual — una arquitectura híbrida poco habitual hoy y útil como recordatorio de que no todo componente tiene que ser aprendido.

---

**Ver también:** [Looking to Listen (2018)](/papers/looking-to-listen-ephrat-2018) · [Separación de Fuentes](/fundamentos/separacion-de-fuentes) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) · [Objects that Sound (2018)](/papers/objects-that-sound-arandjelovic-2018)
