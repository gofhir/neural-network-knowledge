# Actividad 2 — Efecto de WAYS y SHOTS en MAML sobre Omniglot

> Práctico Meta-Aprendizaje (Parte 1) · Diplomado IA UC · Prof. Pablo Messina
> Estudio de la dificultad intrínseca del problema few-shot, usando la mejor
> configuración hallada en la Actividad 1.

## 1. Objetivo

Usando el mejor modelo e hiperparámetros de la Actividad 1
(`OmniglotCNN`, `adaptation_steps=5`, `meta_batch_size=64`, `meta_lr=0.005`,
`fast_lr=0.3`), estudiar cómo varía el desempeño al modificar **WAYS** (número de
clases por tarea) y **SHOTS** (ejemplos de support por clase). El enunciado
recomienda no superar 8 para evitar caídas de Colab por memoria.

Para una comparación justa, **todas las corridas usan el mismo presupuesto de 150
iteraciones**. Lo que importa en este experimento es la *tendencia* relativa entre
configuraciones, no la convergencia total de cada punto.

> Nota técnica: `OmniglotCNN(WAYS)` dimensiona su capa de salida al número de
> clases. Por eso cada corrida **re-instancia el modelo** con el WAYS
> correspondiente; no se reutiliza el mismo objeto entre valores de WAYS.

## 2. Barrido A — efecto de WAYS (SHOTS=1)

| WAYS | Azar (1/N) | Train | Valid | **Test** | Test normalizado* | Convergencia |
|---|---|---|---|---|---|---|
| 2 | 50.0% | 0.977 | 0.922 | **0.932** | **0.864** | aplanada (rápida) |
| 4 | 25.0% | 0.766 | 0.773 | **0.797** | **0.729** | casi aplanada |
| 8 | 12.5% | 0.715 | 0.664 | **0.657** | **0.608** | aún subiendo |

*Test normalizado = `(acc − azar) / (1 − azar)`: fracción del margen sobre el azar
que el modelo logra capturar.

**Observaciones:**

1. **Hipótesis confirmada:** a más WAYS, menor accuracy crudo
   (0.932 → 0.797 → 0.657). Más clases = más fronteras de decisión y más pares
   confundibles, con solo 1 shot de evidencia.
2. **La caída es real, no solo efecto del azar:** al normalizar contra el azar, la
   dificultad sigue subiendo (0.864 → 0.729 → 0.608). WAYS mide dificultad
   intrínseca de discriminación, no solo recalibra el piso aleatorio.
3. **Matiz de subentrenamiento:** los problemas más difíciles convergen más lento.
   WAYS=2 se aplana temprano (le sobran 150 iters), pero WAYS=8 sigue subiendo al
   final → está subentrenado. Con presupuesto fijo de iteraciones, el extremo
   difícil queda penalizado; la magnitud del gap 4→8 está algo inflada por esto
   (con 400 iters el 8-way probablemente subiría a ~0.72-0.78).
4. **Sin overfitting:** en los tres casos train ≈ valid. La configuración
   estabilizada de la Act.1 (`batch=64`, `fast_lr=0.3`) mantiene el entrenamiento
   sano independientemente del WAYS.

## 3. Barrido B — efecto de SHOTS (WAYS=4)

| SHOTS | Train | Valid | **Test** | Error final | Test normalizado |
|---|---|---|---|---|---|
| 1 | 0.766 | 0.773 | **0.797** | 0.136 | 0.729 |
| 5 | 0.921 | 0.946 | **0.943** | 0.009 | 0.923 |

**Salto: 0.797 → 0.943 = +0.146** solo por pasar de 1 a 5 ejemplos de support.

**Observaciones:**

1. **Más evidencia = adaptación más robusta.** Con 5 imágenes por clase, el bucle
   interno de MAML estima mucho mejor cada tarea. El error de query se desploma de
   0.136 → 0.009 (≈15× menor).
2. **El azar no cambia** (sigue 25%, depende solo de WAYS): toda la ganancia es
   poder de adaptación puro.
3. **Convergencia más estable:** curvas suaves, train ≈ valid; más shots reduce la
   varianza de la adaptación interna → meta-gradiente más informativo.

## 4. Cuadro consolidado (ambos ejes)

| Config | Azar | **Test** | Lectura |
|---|---|---|---|
| WAYS=2, SHOTS=1 | 50.0% | 0.932 | pocas clases → fácil |
| WAYS=4, SHOTS=1 | 25.0% | 0.797 | referencia |
| WAYS=8, SHOTS=1 | 12.5% | 0.657 | muchas clases → difícil (subentrenado) |
| WAYS=4, SHOTS=5 | 25.0% | 0.943 | mucha evidencia → fácil |

### Los dos ejes son fuerzas opuestas de magnitud comparable

```
Aumentar WAYS  4→8 (más clases):     0.797 → 0.657   Δ = −0.140   (endurece)
Aumentar SHOTS 1→5 (más evidencia):  0.797 → 0.943   Δ = +0.146   (facilita)
```

### Las palancas se compensan

```
WAYS=2, SHOTS=1  →  0.932   (2 clases, 1 ejemplo)
WAYS=4, SHOTS=5  →  0.943   (4 clases, 5 ejemplos)
```

Un problema de 4 clases con 5 shots es tan resoluble como uno de 2 clases con 1 shot.
La dificultad extra de duplicar las clases se compensa dando más ejemplos de support.
La dificultad real del few-shot es la razón **evidencia / nº de clases**, no cada eje
por separado.

## 5. Conclusiones

1. **WAYS y SHOTS son los dos ejes ortogonales de la dificultad few-shot.** WAYS la
   aumenta (más clases que discriminar + menor azar); SHOTS la reduce (más evidencia
   por clase, sin tocar el azar).
2. El accuracy responde monotónicamente a ambos, con efectos de magnitud comparable
   y signo opuesto.
3. Aun normalizando contra el azar, aumentar WAYS endurece genuinamente el problema:
   no es un mero artefacto del piso aleatorio.
4. Los problemas más difíciles (más WAYS) **convergen más lento** y requieren más
   iteraciones; con presupuesto fijo, su accuracy queda subestimado.
5. La configuración estabilizada de la Act.1 generaliza bien: mantuvo train ≈ valid
   (sin overfitting) en todas las variantes de WAYS y SHOTS.
