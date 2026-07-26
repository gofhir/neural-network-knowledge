---
title: "Common Voice: corpus de voz multilingüe (2020)"
weight: 417
math: true
---

{{< paper-card
    title="Common Voice: A Massively-Multilingual Speech Corpus"
    authors="Rosana Ardila et al. (Mozilla)"
    year="2020"
    venue="LREC 2020 / arXiv:1912.06670"
    pdf="/papers/common-voice-ardila-2020.pdf" >}}
Common Voice es un corpus de voz transcrita **masivamente multilingüe** construido por **crowdsourcing** y liberado bajo licencia **Creative Commons CC0 (dominio público)**. Voluntarios **graban** oraciones leídas en pantalla y **validan** por votación las grabaciones de otros, produciendo pares ⟨audio, transcripción⟩ sin costo de etiquetado. Al momento del paper reúne **29 idiomas activos** (38 recolectando datos), más de **50.000 personas** y **2.500 horas** de audio, lo que sus autores presentan como el mayor corpus de audio en dominio público para reconocimiento del habla (ASR). Como demostración de utilidad, entrenan modelos con DeepSpeech y logran, vía *transfer learning* desde inglés, una mejora promedio de **5.99 ± 5.48** puntos de *Character Error Rate* en doce idiomas objetivo. Para la [Clase 37](/clases/clase-37) es el caso canónico de dataset de escala web con licencia abierta y el eje de **sesgo y representación**.
{{< /paper-card >}}

---

## Contexto: el ASR concentrado en pocos idiomas

Las tecnologías del habla funcionan bien en inglés, mandarín y un puñado de idiomas de **altos recursos**, porque para ellos existen corpus grandes; para los miles de idiomas *low-resource* el dato es prohibitivamente caro o inexistente, y sin dato no hay modelo. El paper enmarca esto como un problema no solo técnico sino de **valores**: la tecnología del habla "debería ser abierta y descentralizada". El problema tiene dos caras: **cobertura** (la investigación se concentra donde ya hay datos) y **licencia** (muchos corpus multilingües no son abiertos). Los antecedentes fallan en al menos un eje: **Babel** es de alta calidad pero cerrado; **VoxForge** es comunitario y abierto pero sin pipeline sostenible ni validación; **M-AILABS** carece del componente comunitario. Common Voice se posiciona como alternativa **sostenible y abierta**, arrancó en inglés en julio de 2017 y se abrió a cualquier idioma en junio de 2018.

## Composición: doble crowdsourcing y metadatos

La contribución es **doble** y es de infraestructura, no de arquitectura: un **corpus CC0** y la **plataforma que lo produce** y lo hace escalar. El proceso tiene dos actos que cualquier voluntario realiza desde el navegador:

- **Grabación.** El contribuyente lee en voz alta oraciones que aparecen en pantalla; puede saltar o reportar una oración problemática. El dato es un par ⟨audio, transcripción⟩ donde el texto se conoce de antemano, lo que hace el etiquetado gratuito y confiable.
- **Validación por votación.** Hasta **tres contribuyentes** escuchan cada clip: dos *up-votes* lo marcan **válido**, dos *down-votes* lo marcan **inválido**. Solo los válidos entran a train/dev/test; los indecisos se publican como "other". Este control por la misma multitud es lo que distingue a Common Voice de VoxForge.

El **particionado evita fuga de hablantes**: un locutor aparece en un solo conjunto, para una evaluación honesta de la generalización a hablantes nuevos. El tamaño de los splits se fija con un análisis de potencia estadística (99% de confianza, 1% de margen de error). El audio se distribuye como **MP3 mono, 16 bits, 48 kHz**: un formato **con pérdida** elegido deliberadamente por ser el más universalmente soportado en la web, un compromiso consciente entre fidelidad y accesibilidad masiva. Cada idioma se descarga con seis archivos TSV cuyas columnas `[client_id, path, sentence, up_votes, down_votes, age, gender, accent]` incluyen **metadatos demográficos autorreportados y opcionales** de edad, género y acento.

## El eje de sesgo y representación

Este es el eje que la Clase 37 pone en primer plano, y el corpus lo aborda de forma explícita y estructural.

- **Desbalance entre idiomas.** La distribución de horas es profundamente desigual: el **inglés** domina con **39.577 voces** y del orden de mil horas validadas, mientras muchos idiomas tienen **menos de una hora**. El **alemán** aporta más de **65.000 clips** frente a **menos de 1.000 para el esloveno**. Esta cola larga reproduce, dentro del propio corpus abierto, el sesgo de altos-vs-bajos recursos que busca combatir; el paper lo compensa parcialmente con *transfer learning*, reconociendo que el dato por sí solo no basta.
- **Representación demográfica.** Las columnas de edad, género y acento **habilitan la auditoría** de representación que la clase exige. Pero al ser opcionales quedan frecuentemente vacías, y la recolección crowdsourced tiende a **sobrerrepresentar el perfil típico del voluntario**: la herramienta para medir el sesgo existe, pero el sesgo de participación persiste —quién decide grabar determina qué voces terminan en el corpus.

El mensaje transversal: **la apertura y la escala no eliminan el sesgo por sí solas**. Un corpus CC0 gigante puede seguir siendo demográficamente estrecho.

## Impacto

Common Voice se volvió **infraestructura fundacional del ASR multilingüe moderno**: su combinación de escala, apertura CC0 y cobertura de idiomas de bajos recursos lo convirtió en fuente estándar para entrenar y evaluar sistemas de habla. Está en el linaje de corpus abiertos que alimentan a modelos posteriores como [Whisper](/papers/whisper-radford-2022) (OpenAI, 2022) y MMS (Meta, 2023, hacia mil idiomas). Más allá de esto, popularizó un **modelo de producción de datasets** —crowdsourcing con validación por pares, dominio público, crecimiento comunitario— replicado por otros proyectos: no un dataset congelado, sino una plataforma que sigue creciendo con cada *release* anual.

## Limitaciones

- **Calidad variable.** Voluntarios con dispositivos y entornos heterogéneos, más compresión MP3, dan audio de calidad desigual; la validación filtra lo peor pero no homogeneiza.
- **Desbalance severo entre idiomas.** Para la mayoría de los idiomas de la cola, el dato es insuficiente para entrenar un ASR competitivo sin transfer learning.
- **Sesgo de participación demográfica.** Los metadatos de edad/género/acento son opcionales y a menudo ausentes; la población de voluntarios no representa a los hablantes reales.
- **Habla leída, no espontánea.** El corpus captura lectura de oraciones, no conversación; los modelos pueden generalizar peor a disfluencias y diálogo real.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) —segunda del módulo sobre audio— trata **de dónde salen los datos** con que se entrenan los sistemas de habla, y Common Voice es su caso de estudio canónico como [dataset de audio](/fundamentos/datasets-de-audio) en tres frentes: **escala web** (crowdsourcing masivo que produce miles de horas sin comprar dato), **sesgo y representación** (la pregunta guía "¿el dataset cubre acentos, géneros e idiomas?" se vuelve operable con las columnas demográficas y el desbalance explícito de la Tabla 1) y **licencia y formato** (CC0 como grado máximo de apertura, MP3 como compromiso práctico).

**Relevancia para salud.** El sesgo de representación no es una curiosidad académica: un ASR clínico entrenado sobre datos que sobrerrepresentan ciertos acentos, géneros y edades **fallará justo con los pacientes más vulnerables** —hablantes de lenguas minoritarias, migrantes, adultos mayores, poblaciones rurales—. La equidad en salud digital exige preguntar, antes de desplegar, lo que la Clase 37 enseña a preguntar: *¿este dataset se parece a la población real de pacientes que va a atender?*
