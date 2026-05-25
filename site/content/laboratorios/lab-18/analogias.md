---
title: "Bloque 1 — Analogías con 3CosMul"
weight: 10
math: true
---

Recorrido del bloque de analogías (Celdas 10-23 del notebook). El lab te entrega 7 ejemplos guía y te pide generar 3 propios + comentario crítico (Actividad 1).

## La función `most_similar_cosmul`

Implementa la **fórmula (4) de [Levy & Goldberg 2014 CoNLL](/papers/linguistic-regularities-levy-goldberg-2014)**:

$$
b^* = \arg\max_{x \in V} \frac{\cos_+(x, b) \cdot \cos_+(x, a^*)}{\cos_+(x, a) + \varepsilon}
$$

donde $\cos_+(u,v) = (\cos(u,v)+1)/2 \in [0, 1]$ y $\varepsilon = 0.001$.

**Sintaxis del notebook**:

```python
google_wordvecs.most_similar_cosmul(positive=[a, b_p], negative=[a_p])
# pregunta: "a_p es a a como b_p es a ???"
```

## Notación

| Notación lab (Pablo Messina) | Notación paper (Levy-Goldberg) | Rol |
|---|---|---|
| $a$ | $b$ | Atractor 1 |
| $b_p$ | $a^*$ | Atractor 2 |
| $a_p$ | $a$ | Repulsor |
| $b$ (incógnita) | $b^*$ | Resultado |

## Los 7 ejemplos guía del lab

| # | Analogía | Top-1 | Score | Gap top1-top2 | Observación |
|---|---|---|---|---|---|
| 1 | `woman+king−queen` (man) | man | 0.93 | 0.10 | Ruido de prensa policial (motorcyclist, taxi_driver) |
| 2 | `actor+woman−man` (actress) | **actress** | **1.06** | 0.13 | Top-10 100% cine, aparecen Paltrow/Heigl |
| 3 | `son+woman−man` (daughter) | **daughter** | 1.05 | **0.04** | Husband/father intrusos (cross-género por co-ocurrencia) |
| 4 | `play+singing−sing` (playing) | playing | 0.92 | 0.10 | Intoxicación deportiva (game, scoring) |
| 5 | `play+sang−sing` (played) | played | 0.91 | 0.05 | No generaliza "pasado verbal" como eje universal |
| 6 | `Santiago+Venezuela−Chile` (Caracas) | Caracas | 0.95 | **0.013** | 8/10 apellidos hispanos (Santiago = apellido) |
| 7 | `Buenos_Aires+Chile−Santiago` (Argentina) | **Argentina** | 1.01 | 0.09 | Top-10 100% sudamericano (negación cancela polisemia) |

## Patrones empíricos identificados

### 1. **Polisemia ortográfica** (caso Santiago)

`Santiago` aparece en prensa tanto como **capital chilena** como **apellido hispano frecuente**. Su vector está dominado por el cluster de apellidos. Cuando se usa como atractor positivo, contamina el top-10.

```python
positive=['Santiago', 'Venezuela'], negative=['Chile']
# Top-10: Caracas (✅) + Ramírez, Peña, Chavez, Rodríguez,
#         Martinez, Ramirez, Ortiz, Venezuelan, Martínez
```

→ **8 de 10 resultados son apellidos hispanos**, no ciudades venezolanas. Gap top1-top2 = 0.013 (al borde de fallar).

### 2. **El "soft-or" cuando la palabra polisémica está en `positive`**

Levy-Goldberg (Sección 6) advierten: 3CosAdd sufre soft-or porque una similitud dominante aplasta las demás. 3CosMul lo mitiga pero no lo elimina cuando la polisemia es severa.

**Comparación dual**:

- Cuando `Santiago` está en `positive` (Celda 17) → contaminación masiva.
- Cuando `Santiago` está en `negative` (Celda 18) → su polisemia se **aprovecha** para cancelar clusters secundarios. Top-10 limpio.

→ **Lección práctica**: usar palabras polisémicas como **repulsores**, no atractores.

### 3. **Sesgo temporal del corpus**

Google News fue entrenado pre-2013. Eventos posteriores no existen, eventos previos con larga cobertura dominan:

| Pregunta | Aparece | NO aparece |
|---|---|---|
| Top-10 para Venezuela | **Chavez** (presidente 1999-2013) | Maduro (asume 2013) |
| Top-10 para Argentina | **Nestor_Kirchner** (2003-2007) | Macri (2015), Milei (2023) |
| Microsoft + iPhone − Apple | **Windows_Mobile** (descontinuado 2010, 7 años cobertura) | Surface (2012, recién lanzado) |

→ El embedding refleja el **peso ACUMULADO de cobertura mediática**, no el estado vigente.

## Experimentos diagnósticos propios

### Experimento 1 — ¿Existe un "eje de pasado verbal" universal?

Calculé la similitud coseno entre las direcciones `verbo_pasado − verbo_infinitivo` para 4 verbos:

```
cos(ate-eat, walked-walk) = 0.486
cos(ate-eat, ran-run)     = 0.395
cos(ate-eat, studied-study) = 0.187
cos(ran-run, walked-walk) = 0.597
cos(ran-run, studied-study) = 0.153
cos(studied-study, walked-walk) = 0.245
```

**Rango**: 0.15-0.60, **promedio**: 0.35. No hay un "eje pasado" universal. Las direcciones varían según el verbo, reflejando que **Word2Vec opera por asociaciones léxicas locales**, no por categorías gramaticales abstractas.

### Experimento 2 — ¿Existe un "eje capital → país"?

Sobre 5 capitales monosémicas (Tokyo, Caracas, Stockholm, Helsinki, Brussels):

```
cos(Caracas-Venezuela, Tokyo-Japan)      = 0.657
cos(Stockholm-Sweden, Tokyo-Japan)       = 0.699
cos(Helsinki-Finland, Tokyo-Japan)       = 0.651
cos(Helsinki-Finland, Stockholm-Sweden)  = 0.719  ← máximo
cos(Brussels-Belgium, *)                 = 0.40-0.49  ← outlier
```

**Rango**: 0.40-0.72, **promedio**: 0.56. Más estructurado que el eje verbal, pero **Brussels actúa como outlier** por **polisemia funcional**: en prensa anglosajona aparece dominantemente como "sede de la UE" más que como "capital nacional".

## Las 3 analogías propias (Actividad 1)

| # | Dominio | Analogía | Top-1 | Score | Hallazgo |
|---|---|---|---|---|---|
| 1 | Música | `saxophone+classical−jazz` | **cello** | 0.86 | Sitar/tabla aparecen → **polisemia cultural** (classical occidental + india) |
| 2 | Deportes | `Yankees+basketball−baseball` | **Knicks** | 0.88 | Lakers NO aparece → modelo capturó dimensión geográfica (NY/costa este), no éxito |
| 3 | Tecnología | `Microsoft+iPhone−Apple` | **Windows_Mobile** | 0.86 | 4/10 OS móviles MS → analogía funciona, pero sesgo legacy > vigente |

## Cross-links

{{< cards >}}
  {{< card link="../" title="← Lab 18 - Hub" subtitle="Volver al índice del lab" icon="academic-cap" >}}
  {{< card link="../doesnt-match" title="Bloque 2 - doesnt_match →" subtitle="Detección de outliers por centroide" icon="academic-cap" >}}
  {{< card link="/papers/linguistic-regularities-levy-goldberg-2014" title="Paper 3CosMul" subtitle="Levy-Goldberg 2014 CoNLL - fórmula (4)" icon="document-text" >}}
  {{< card link="/papers/contrastive-analogies-ri-lee-verma-2023" title="Paper Teorema 1" subtitle="Líneas paralelas con ζ" icon="document-text" >}}
{{< /cards >}}
