---
title: "DAgger: Dataset Aggregation para Imitación (2011)"
weight: 372
math: true
---

{{< paper-card
    title="A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
    authors="Stéphane Ross, Geoffrey Gordon, J. Andrew Bagnell (CMU)"
    year="2011"
    venue="AISTATS 2011"
    pdf="/papers/dagger-ross-2011.pdf" >}}
Este es el paper que diagnostica y **cura** el fallo estructural del aprendizaje por imitación ingenuo (el *behavioral cloning* o clonación de comportamiento). Cuando se entrena un clasificador para imitar a un experto usando solo sus demostraciones, las predicciones del aprendiz alteran las observaciones futuras y violan la suposición i.i.d. del aprendizaje supervisado: los errores **se componen** y una política con tasa de error $\epsilon$ bajo la distribución del experto puede cometer hasta $O(T^2\epsilon)$ errores bajo la distribución que ella misma induce. **DAgger** (*Dataset Aggregation*) reduce esa cota a $O(T\epsilon)$ —lineal en el horizonte $T$— con un meta-algoritmo notablemente simple y sin parámetros libres, interpretando la imitación como una **reducción a aprendizaje online sin arrepentimiento** (*no-regret*). Es el algoritmo que implementa el [laboratorio de la Clase 33](/laboratorios/lab-33) sobre Atari Breakout con un experto DQN.
{{< /paper-card >}}

---

## Contexto: por qué falla el behavioral cloning

El enfoque tradicional de la [imitación](/fundamentos/aprendizaje-por-imitacion) reduce el problema a aprendizaje supervisado: se recolectan pares (observación, acción experta) mientras el experto conduce el sistema y se entrena un clasificador que prediga la acción de $\pi^*$ dada la observación. El problema es que la distribución de estados depende de la política que se ejecuta. Sea $d_\pi$ la distribución promedio de estados al seguir $\pi$ durante $T$ pasos y $\ell(s,\pi)$ una pérdida sustituta (por ejemplo la 0-1 respecto de $\pi^*$). El objetivo *deseable* es

$$\hat\pi = \arg\min_{\pi\in\Pi}\mathbb{E}_{s\sim d_\pi}[\ell(s,\pi)],$$

es decir, minimizar la pérdida bajo la **propia distribución inducida**. Pero $d_\pi$ depende de $\pi$ y no se puede calcular, solo muestrear. El *behavioral cloning* ignora este acoplamiento y minimiza la pérdida bajo la distribución del **experto** $d_{\pi^*}$: optimiza la distribución equivocada.

El fallo tiene dos causas entrelazadas. Primero, el **compounding error**: apenas el aprendiz comete un error se desvía de la trayectoria del experto y encuentra observaciones distintas a las de la demostración; cada nuevo error lo empuja a regiones más ajenas y los errores se acumulan multiplicativamente. Segundo, los **estados fuera de distribución**: el aprendiz nunca vio en entrenamiento los estados que visita tras equivocarse —porque el experto, siendo bueno, nunca los visita— así que no aprendió a **recuperarse** de sus propios errores. El clasificador puede tener baja pérdida sobre $d_{\pi^*}$ y ser desastroso sobre $d_{\hat\pi}$.

La cuantificación formal (Teorema 2.1, Ross y Bagnell 2010) es contundente: si $\mathbb{E}_{s\sim d_{\pi^*}}[\ell(s,\pi)] = \epsilon$ con $\ell$ cota superior de la pérdida 0-1, entonces

$$J(\pi) \le J(\pi^*) + T^2\epsilon.$$

La cota es **cuadrática en el horizonte** y es *ajustada* (tight): existen ejemplos —el de Kääriäinen (2006) en predicción de secuencias— donde el costo escala como $\Theta(T^2)$. Duplicar el horizonte cuadruplica el costo del error: catastrófico para tareas largas.

## Método: DAgger y la reducción a no-regret

DAgger construye iterativamente el conjunto de estados que la política aprendida encontrará realmente en ejecución. En cada iteración: (a) ejecuta la política actual para recolectar los estados que **efectivamente visita**, (b) consulta al experto qué acción tomaría en **esos** estados, (c) **agrega** esos pares al conjunto de datos acumulado y (d) reentrena sobre todo el conjunto agregado.

```
Inicializar D ← ∅;  π̂₁ ← cualquier política en Π.
para i = 1 hasta N:
    πᵢ = βᵢ·π* + (1 − βᵢ)·π̂ᵢ          # mezcla experto/aprendiz
    Muestrear trayectorias de T pasos usando πᵢ.
    Dᵢ = {(s, π*(s))} : estados VISITADOS por πᵢ.
    D ← D ∪ Dᵢ.                        # agregación
    Entrenar π̂ᵢ₊₁ sobre D.
Retornar la mejor π̂ᵢ según validación.
```

El punto crucial es la línea de las etiquetas: **el experto etiqueta los estados que visita el aprendiz, no los que visitaría él mismo.** Ahí está el arreglo del *distribution shift*: se le pregunta al experto "¿qué harías tú en el lío en el que yo me metí?". La política de mezcla $\pi_i = \beta_i\pi^* + (1-\beta_i)\hat\pi_i$ cede el control al experto una fracción $\beta_i$ del tiempo; se toma $\beta_1=1$ (la primera iteración es pura demostración experta, equivalente al dataset de BC) y el único requisito teórico es que el promedio $\bar\beta_N\to 0$. La versión sin parámetros que suele rendir mejor es $\beta_i=\mathbb{I}(i=1)$.

La garantía surge de interpretar DAgger como *Follow-The-Leader* sobre las pérdidas online $\ell_i(\pi) = \mathbb{E}_{s\sim d_{\pi_i}}[\ell(s,\pi)]$. **Cualquier** algoritmo no-regret aplicado así encuentra una política con buen desempeño bajo su propia distribución inducida. Un algoritmo es no-regret si su arrepentimiento promedio $\gamma_N\to 0$; para pérdidas fuertemente convexas, FTL garantiza $\gamma_N=\tilde{O}(1/N)$. El resultado central (Teorema 3.2), combinando con la cota del sobrecosto de un error $u$, es

$$J(\hat\pi) \le J(\pi^*) + uT\epsilon_N + O(1),$$

crecimiento **lineal** en $T$ frente al $T^2\epsilon$ del behavioral cloning, siempre que exista una buena política en la clase bajo la distribución agregada y que el experto se recupere bien de los errores ($u=O(1)$).

## Resultados

DAgger se validó en tres problemas. En **Super Tux Kart** (conducción 3D con regresión lineal ridge), el supervisado no mejora aunque se recolecten más datos —nunca enseña a recuperarse— mientras DAgger logra una política que **nunca se cae** de la pista tras 15 iteraciones. En **Super Mario Bros.** (experto = planificador casi óptimo, aprendiz = 4 SVM lineales), el supervisado se **estanca** porque Mario queda atascado contra obstáculos que el experto siempre salta con antelación y nunca demuestra cómo desatascar; DAgger supera a SMILe y SEARN, y usar al experto una fracción pequeña del tiempo ($\beta_i=0.5^{i-1}$) rinde ligeramente mejor porque desatasca a Mario y recolecta datos más variados. En **reconocimiento de escritura** (predicción estructurada como imitación degenerada), DAgger sube a **85.5%** frente al 83.6% del supervisado y el 82% de la base sin estructura.

## Limitaciones

- **Requiere un experto interactivo consultable durante el entrenamiento.** Es la limitación práctica más importante: hay que poder preguntar $\pi^*(s)$ para **cualquier** estado que el aprendiz visite, no solo sobre trayectorias grabadas. Eso es costoso o inviable con un experto humano, o peligroso si visitar ciertos estados en el mundo real tiene consecuencias (un auto que "practica" caerse por un barranco).
- **Supuesto de convexidad fuerte / no-regret** y **realizabilidad implícita:** la cota se vacía si la clase $\Pi$ no contiene una política capaz de imitar bien al experto sobre los estados agregados.
- **Dependencia de $u$:** si el costo de la tarea no coincide con la pérdida sustituta, $u$ puede ser $O(T)$ en el peor caso y devolver el crecimiento cuadrático.
- **Muestras finitas exigen bastantes iteraciones:** $N$ del orden $O(T^2\log(1/\delta))$ en el análisis básico, aunque la convexidad fuerte lo mejora a $\tilde{O}(T\log(1/\delta))$.

## Por qué importa para la Clase 33

DAgger es el **corazón teórico** de la [Clase 33](/clases/clase-33). La clase presenta el [aprendizaje por imitación](/fundamentos/aprendizaje-por-imitacion) en dos niveles: primero el *behavioral cloning* como punto de partida natural y su **fallo diagnosticado** —los dos motivos, compounding error y estados fuera de distribución, son literalmente los que este paper formaliza con la cota $O(T^2\epsilon)$—; y luego **DAgger** (slide 36) como la corrección, con la garantía lineal $O(T\epsilon)$ que se sigue de la reducción a no-regret.

El [laboratorio de la clase](/laboratorios/lab-33) implementa DAgger sobre **Atari Breakout con un experto DQN**. El mapeo es directo: la política aprendida es una red que predice la acción a partir de la imagen del juego, y el experto $\pi^*$ es un agente [DQN](/papers/dqn-nature-mnih-2015) preentrenado (mismo linaje de la [Clase 31](/clases/clase-31)) que se puede consultar en cualquier estado —justo lo que DAgger necesita y un humano no podría dar. El lab hace visible la diferencia entre entrenar sobre $d_{\pi^*}$ (BC, que se descarrila cuando el aprendiz se mete en configuraciones de ladrillos que el DQN nunca dejó ocurrir) y entrenar sobre la distribución **agregada** de estados que el propio aprendiz visita.

DAgger es el eslabón que hace explícito *por qué* la distribución de datos —y no solo la calidad del clasificador— es el problema. La trayectoria de la clase se lee como: BC (falla por *distribution shift*) → **DAgger** (arregla el shift consultando al experto interactivo) → [IRL](/fundamentos/aprendizaje-reforzado-inverso) (recupera la recompensa, más transferible) → [GAIL](/papers/gail-ho-ermon-2016) (empareja distribuciones adversarialmente, sin experto interactivo).
