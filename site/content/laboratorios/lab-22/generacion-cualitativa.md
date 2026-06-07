---
title: "P2 — Generación cualitativa (Actividad 2)"
weight: 6
---

> **Celdas 16-27 del notebook (Parte 2).** Probar T5 con textos de la "vida real" — una noticia y párrafos de un libro — para intuir sus capacidades y fallos antes de medir con ROUGE.

## Caso 1: noticia real (DermaSensor / FDA)

Se resumió una noticia de salud (la FDA aprueba DermaSensor, primer dispositivo de IA para diagnosticar cáncer de piel) con `num_beams=20`, `num_return_sequences=5`.

### Resultados y análisis

**Saliencia a medias:** los 5 resúmenes lideran con *"the first AI-powered device to diagnose skin cancer noninvasively at the point of care"* — el hecho correcto. Pero **ninguno menciona "FDA" ni "approved"**: el modelo se comió el sujeto y el verbo principal, quedándose con la aposición que describe al dispositivo. Capturó *qué es*, perdió *qué pasó*.

**Eliminó TODAS las cifras** (el hallazgo central): la noticia tenía 96% sensibilidad, 224 tipos, 97% benignos. **Ningún resumen los incluyó** — los 5 saltaron de la oración 1 a la 5, omitiendo las oraciones de números.

> **Caso opuesto al de COVID.** Allá el modelo se *aferró* a las cifras; aquí las *descartó*. No hay consistencia en cómo trata los datos numéricos — depende de la estructura de la oración que los contiene. **En contexto clínico esto es peligroso:** un resumen de un diagnóstico de cáncer que omite la sensibilidad del 96% eliminó justo el dato que un médico necesita. Nota positiva: al omitir, tampoco alucinó.

**Diversidad del beam:** 4 de los 5 fueron clones (variaron "earlier"/"sooner", presencia de "an"/coma). El **Resumen 4 divergió** (arrancó nombrando "DermaSensor" y describió la spectroscopy). Con `num_beams=20`, los beams del fondo del ranking pueden tomar bifurcaciones distintas.

**Abstracción real detectada:** el Resumen 2 usó "sooner" (sinónimo de "earlier", no en el original); el Resumen 4 escribió "analyze" (el original decía "analyzes"). Generación genuina, pero leve.

### Caso 2: párrafos de un libro (*Pride and Prejudice*)

Se resumieron los párrafos iniciales del Capítulo 1 de Jane Austen — un texto **out-of-distribution** para un modelo entrenado en noticias.

**El modelo no entendió de qué trata el texto.** El fragmento cuenta que la Sra. Bennet anuncia que Netherfield Park fue alquilado. El resumen produjo una **ensalada de fragmentos inconexos**:
- "little known the feelings or views of such a man..." — un trozo del medio, **sin sujeto**.
- "he is considered the rightful property of some one or other of their daughters" — otro trozo.

**El hecho central —Netherfield Park alquilado— desapareció.**

**Alucinación de atribución (el fallo más grave):** los resúmenes 2 y 3 convirtieron *"said his lady to him"* en *"she tells him"* — una abstracción sofisticada (resolvió la correferencia "his lady" → "she", cambió el tiempo verbal). Pero la usó para **atribuir a la Sra. Bennet una frase que es narración irónica de la autora**. Gramaticalmente impecable, factualmente inventado.

> **El fallo más insidioso de los abstractivos, y el más relevante para dominios críticos:** el texto suena coherente y bien construido, pero afirma algo que el original no dice. Un lector que no conozca el texto lo creería. En un resumen clínico sería atribuir un síntoma al paciente equivocado — fluido, plausible y falso.

**Truncamiento degenerado:** el Resumen 4 terminó en *"on his first."* — cortó la frase a media idea. El `early_stopping` detuvo el beam en un punto gramaticalmente inválido. Señal de que el modelo está perdido fuera de su dominio.

## Comparación de los dos casos

| | Noticia (DermaSensor) | Libro (Pride and Prejudice) |
|---|---|---|
| Hecho principal | Parcial (perdió "FDA approved") | **Perdido** |
| Coherencia | Alta | **Fragmentaria** |
| Alucinación | No | **Sí (atribución falsa)** |
| Truncamiento | No | **Sí (Resumen 4)** |
| Calidad global | Aceptable | **Mala** |

> **La lección de la Actividad 2:** un modelo de resumen **no es de propósito general** — es tan bueno como el dominio en que fue fine-tuneado. T5-small en noticias → razonable en noticias, inservible en literatura. Y el modo de fallo fuera de dominio no es "quedarse callado", es **producir texto fluido pero incorrecto**. Por eso en un dominio crítico no se puede tomar un modelo genérico y aplicarlo a ciegas: fallaría *con confianza*. La mejora más directa (sin tocar la decodificación) es **fine-tunear el modelo** sobre el tipo de texto objetivo, o usar uno más grande ([BART](/papers/bart-lewis-2020)/[PEGASUS](/papers/pegasus-zhang-2020)).

---

**Anterior:** [abstractivo con T5](abstractivo-t5) · **Siguiente:** [parámetros de decodificación (Act. 3)](decodificacion)
