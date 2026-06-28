# Actividad 3 — MAML sobre Mini-ImageNet: información vs optimización

> Práctico Meta-Aprendizaje (Parte 1) · Diplomado IA UC · Prof. Pablo Messina
> Estudio de las tres palancas de mejora (argumentos, SHOTS, WAYS) en el dataset
> difícil, y comparación con el comportamiento observado en Omniglot.

## 1. Punto de partida

El baseline del notebook (`MiniImagenetCNN`, 4-way 1-shot, 60 iteraciones) entrega:

```
Meta Test Accuracy = 0.299   (azar = 25%)
```

Apenas por encima del azar. Contraste con el mismo problema en Omniglot
(4-way 1-shot, baseline FC 50 iters): **0.699**. El enunciado lo anticipa:
Mini-ImageNet es mucho más difícil por la naturaleza de las imágenes (fotos
naturales, variación intra-clase enorme, fondos distractores).

**Fenómeno notable del baseline:** el error (cross-entropy) baja de 2.5 → 0.4, pero
el accuracy se queda plano y ruidoso en ~0.30. El modelo aprende a no estar
"confiado-equivocado" (baja la loss) pero no logra discriminar las clases (accuracy
estancado). Es la firma de un modelo subentrenado en un problema difícil.

## 2. Diseño experimental

Se respondieron las tres preguntas del enunciado con un ablation, todas las corridas
a 150 iteraciones, `adaptation_steps=5`, `meta_lr=0.005`, `fast_lr=0.3`,
`meta_batch_size=32` (no se subió a 64 por el alto consumo de memoria de
Mini-ImageNet). El modelo se re-instancia con `MiniImagenetCNN(WAYS)` en cada corrida.

| # | Config | Test crudo | Azar | **Test normalizado*** | Lectura |
|---|---|---|---|---|---|
| base | 4w1s · 60 iters | 0.299 | 25% | 0.065 | ≈ azar |
| 1 | 4w1s · 150 + args | 0.324 | 25% | **0.099** | argumentos: poco |
| 2 | 4w5s · 150 + args | 0.491 | 25% | **0.321** | SHOTS: la ganadora |
| 3 | 2w1s · 150 + args | 0.547 | 50% | **0.094** | WAYS solo: espejismo del azar |
| 4 | 2w5s · 150 + args | 0.710 | 50% | **0.420** | combo: mejor de todos, sin overfitting |

*Test normalizado = `(acc − azar) / (1 − azar)`: fracción del margen sobre el azar
capturada. Es la métrica justa para comparar configuraciones con distinto número de
clases.

### Diseño factorial 2×2 (normalizado)

| | SHOTS=1 | SHOTS=5 |
|---|---|---|
| **WAYS=4** | 0.099 | 0.321 |
| **WAYS=2** | 0.094 | 0.420 |

**Efectos:**
```
SHOTS (1→5):  a WAYS=4: +0.222   |  a WAYS=2: +0.326    → dominante siempre
WAYS  (4→2):  a SHOTS=1: −0.005  |  a SHOTS=5: +0.099   → depende de SHOTS (interacción)
```

**Interacción descubierta:** WAYS↓ solo aporta poder discriminativo real cuando hay
evidencia suficiente. Con 1 shot el efecto es nulo (espejismo del azar); con 5 shots
es genuino (+0.099 normalizado). Las palancas no son independientes: reducir clases
solo rinde si el modelo tiene información que explotar.

**Overfitting (brecha train−valid):** 4w1s 0.086 · 4w5s 0.095 · 2w1s 0.078 ·
**2w5s 0.007**. La combinación menos-clases + más-evidencia es la única que elimina
el meta-overfitting que SHOTS por sí solo no curaba.

## 3. Análisis por palanca

**3.1 Mejores argumentos (pregunta 3) — Δ ≈ +0.025 crudo (poco).**
Subir iteraciones de 60 a 150 y afinar lr/adaptation_steps apenas movió el resultado
(0.299 → 0.324). Además apareció overfitting (train 0.391 vs valid 0.305). A
diferencia de Omniglot, donde las iteraciones eran la palanca dominante, aquí la
optimización no es el cuello de botella: con más entrenamiento el modelo memoriza las
clases de train en vez de generalizar.

**3.2 SHOTS↑ (pregunta 1) — Δ ≈ +0.167 crudo, ×3.2 normalizado (la ganadora).**
Pasar de 1 a 5 shots disparó el test de 0.324 → 0.491, y el normalizado de 0.099 →
0.321. Con 5 fotos de support por clase el modelo puede promediar la variación
intra-clase (5 ejemplos distintos en vez de 1) y discriminar de verdad. Confirma que
el cuello de botella en Mini-ImageNet es la **información por tarea**, no la
optimización.
*Matiz honesto:* SHOTS NO redujo el overfitting (brecha train−valid se mantuvo en
~0.09). Más evidencia resuelve la falta de información por tarea, pero no el
meta-overfitting a las clases de train (son dos problemas distintos).

**3.3 WAYS↓ (pregunta 2) — Δ ≈ +0.223 crudo, pero ~0 normalizado (espejismo).**
Bajar WAYS de 4 a 2 subió el accuracy crudo de 0.324 → 0.547, aparentando ser la
mejor palanca. Pero normalizado contra el azar, 4w1s (0.099) y 2w1s (0.094) son
prácticamente iguales: **el poder discriminativo real no mejoró**. El alza es
artefacto del azar subiendo de 25% a 50%. En Mini-ImageNet 1-shot el modelo está tan
limitado por la falta de evidencia que reducir clases no le ayuda a discriminar.

## 4. Comparación con Omniglot (Actividades 1-2)

| | Omniglot | Mini-ImageNet |
|---|---|---|
| Cuello de botella | optimización (subentrenamiento) | información (evidencia por clase) |
| Palanca dominante | iteraciones | SHOTS |
| WAYS↓ normalizado | mejora (0.729 → 0.864) | no mejora (~0.095 constante) |
| Más iteraciones | gran mejora (0.71 → 0.88) | casi nada (+0.025) + overfitting |

## 5. Conclusiones

1. **SHOTS es la única palanca que mejora el poder discriminativo real en
   Mini-ImageNet.** Aumentar la evidencia por clase ataca el cuello de botella de
   fondo (falta de información por tarea).
2. **WAYS↓ mejora solo en apariencia:** sube el accuracy crudo porque sube el azar,
   no porque el modelo clasifique mejor. Hay que normalizar para no engañarse.
3. **Mejores argumentos (iteraciones, lr) aportan poco** en Mini-ImageNet 1-shot, al
   revés que en Omniglot. El problema no es de optimización.
4. **La naturaleza del dato determina qué palanca importa.** En datos simples
   (Omniglot) manda la optimización; en datos complejos (Mini-ImageNet) manda la
   información. No hay receta universal de hiperparámetros.
5. El overfitting a las clases de train persiste en Mini-ImageNet y no se cura con
   SHOTS; requeriría más diversidad de tareas (`num_tasks`, fuera del alcance de los
   argumentos) o regularización explícita.
