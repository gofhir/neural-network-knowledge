---
title: "AV-HuBERT: representaciones audiovisuales autosupervisadas (2022)"
weight: 463
math: true
---

{{< paper-card
    title="Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction"
    authors="Bowen Shi, Wei-Ning Hsu, Kushal Lakhotia, Abdelrahman Mohamed (Meta AI / TTI-Chicago)"
    year="2022"
    venue="ICLR 2022 / arXiv:2201.02184"
    arxiv="2201.02184"
    pdf="/papers/av-hubert-shi-2022.pdf" >}}
El cierre moderno de la línea que abre la Clase 43. En vez de destilar de una modalidad a otra ([SoundNet](/papers/soundnet-aytar-2016)) o fusionarlas para clasificar ([E2E-AVSR](/papers/e2e-avsr-petridis-2018)), AV-HuBERT **enmascara ambos flujos y predice unidades latentes descubiertas y refinadas automáticamente**, al estilo BERT. El resultado que resume el arco completo: **32,5 % de WER en LRS3 usando solo 30 horas etiquetadas**, superando a un sistema previo entrenado con **31 000 horas** — mil veces más datos transcritos. Con las 433 horas completas y auto-entrenamiento baja a **26,9 %**. Y la representación audiovisual mejora también el reconocimiento **solo de audio**: 40 % de reducción relativa del WER, de 2,3 % a 1,3 %.
{{< /paper-card >}}

---

## El método

HuBERT (para audio) aprende enmascarando parte de la entrada y prediciendo **unidades discretas** obtenidas por *clustering* de las representaciones, en un proceso iterativo: se agrupa, se entrena a predecir esos grupos, se re-agrupa con las representaciones mejoradas, y se repite.

AV-HuBERT lo extiende a dos flujos. La entrada es video **y** audio; se enmascaran segmentos de ambos, y el modelo predice las unidades latentes multimodales de las regiones ocultas. Al tener que reconstruir el audio enmascarado desde el video visible y viceversa, la red queda forzada a construir una representación **compartida** entre modalidades.

{{< concept-alert type="clave" >}}
Los tres papers de esta línea usan la misma propiedad —imagen y sonido describen la misma escena— con tres arquitecturas de supervisión distintas:

| | Fuente de la señal | Dirección | Necesita |
|---|---|---|---|
| **SoundNet** (2016) | distribuciones del maestro visual | visión → sonido | un maestro visual entrenado |
| **L³ / correspondencia** (2017) | ¿corresponden estos dos fragmentos? | simétrica | nada preentrenado |
| **AV-HuBERT** (2022) | unidades latentes de lo enmascarado | simétrica | nada preentrenado |

La progresión es de una supervisión externa (el maestro define lo correcto) a una interna (la estructura de los propios datos lo define). Ver [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual).
{{< /concept-alert >}}

## El resultado que importa

Sobre LRS3, el mayor benchmark público de lectura de labios (433 horas):

| Sistema | Datos etiquetados | WER |
|---|---|---|
| Estado del arte previo (Makino et al., 2019) | 31 000 h | 33,6 % |
| **AV-HuBERT** | **30 h** | **32,5 %** |
| AV-HuBERT + auto-entrenamiento | 433 h | **26,9 %** |

Reducir en tres órdenes de magnitud la anotación necesaria y aun así ganar es el argumento entero del preentrenamiento autosupervisado, en una sola fila.

Y hay un resultado colateral que suele pasarse por alto: usar la representación **audiovisual** para reconocimiento de habla **solo de audio** reduce el WER un 40 % relativo (2,3 % → 1,3 %). Haber aprendido con el video presente mejora al modelo incluso cuando el video ya no está — porque los labios ayudaron a desambiguar durante el preentrenamiento.

## Por qué importa para la Clase 43

La [Clase 43](/clases/clase-43) termina en 2018, con un sistema que clasifica 500 palabras aisladas y declara tres limitaciones: vocabulario cerrado, entrenamiento complejo y mala generalización al largo de la secuencia. AV-HuBERT las resuelve las tres, y lo hace **cambiando de dónde viene la supervisión** en vez de agrandar la arquitectura.

Es también la confirmación de la apuesta de [SoundNet](/papers/soundnet-aytar-2016) seis años antes: que la señal que faltaba no eran etiquetas, sino la correspondencia entre modalidades que el video ya trae gratis.

---

**Ver también:** [SoundNet (2016)](/papers/soundnet-aytar-2016) · [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [LipNet (2016)](/papers/lipnet-assael-2016) · [BERT](/fundamentos/bert) · [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) · [Lectura de Labios](/fundamentos/lectura-de-labios)
