---
title: "Modelo de Bradley-Terry"
weight: 287
math: true
---

El **modelo de Bradley-Terry** (Bradley & Terry, *Biometrika* 1952) es el modelo probabilistico clasico para **comparaciones por pares**: dadas dos opciones $y_w$ e $y_l$ con "fortalezas" latentes $r_w$ y $r_l$, la probabilidad de que $y_w$ sea preferida sobre $y_l$ es la sigmoide de la **diferencia** de fortalezas. Es la base estadistica de los rankings deportivos (Elo, TrueSkill), de los sistemas de recomendacion via comparaciones, y -- mas recientemente -- el supuesto **central** sobre el cual se construye [RLHF](/fundamentos/sft) y [DPO](/fundamentos/dpo).

Si quieres entender por que la **loss DPO tiene la forma $-\log\sigma(\beta(\log\pi_w/\pi_{\text{ref},w} - \log\pi_l/\pi_{\text{ref},l}))$**, la respuesta empieza aqui: es Bradley-Terry con el reward sustituido por log-ratios de policy.

---

## 1. Historia: 1952, paired comparisons en estadistica

Ralph Bradley y Milton Terry publicaron en 1952 un metodo para estimar fortalezas a partir de comparaciones por pares. El problema motivador clasico: tenemos $n$ vinos y un panel de catadores; cada catador prueba **pares** de vinos y dice cual prefiere. ¿Como construimos un ranking total a partir de esos juicios pareados, posiblemente inconsistentes?

La idea: a cada item $y_i$ le asignamos una fortaleza latente $r_i \in \mathbb{R}$, y modelamos la probabilidad de que $y_i$ gane contra $y_j$ como funcion creciente de la diferencia $r_i - r_j$. Bradley y Terry eligieron la **logistica** -- una eleccion que despues se mostro equivalente al modelo Elo de ajedrez (Arpad Elo, 1960s) y a la regresion logistica binaria.

Hoy el mismo modelo opera en el corazon de:

- **Elo** y sus descendientes (chess.com, FIDE, Glicko, TrueSkill).
- **Ranking de productos** via A/B testing pareado.
- **Reward models de RLHF** (InstructGPT, ChatGPT, Llama-Chat).
- **DPO** y sus variantes (IPO, KTO, ORPO).

---

## 2. La formula

Para dos opciones $y_w$ (winner / chosen) y $y_l$ (loser / rejected), Bradley-Terry asume:

$$
P(y_w \succeq y_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \sigma(r_w - r_l)
$$

donde $\sigma(z) = 1/(1+e^{-z})$ es la sigmoide. La segunda igualdad es algebra:

$$
\frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \frac{1}{1 + e^{-(r_w - r_l)}} = \sigma(r_w - r_l).
$$

Tres propiedades clave:

1. **Solo importa la diferencia** $r_w - r_l$. Sumar una constante a todas las fortalezas no cambia ninguna probabilidad. El modelo es invariante a translaciones.
2. **Probabilidad simetrica**: $P(y_w \succeq y_l) + P(y_l \succeq y_w) = 1$, sin empates explicitos.
3. **Transitividad estocastica**: si $r_a > r_b > r_c$, entonces $P(a \succeq c) > P(a \succeq b)$ y $P(a \succeq c) > P(b \succeq c)$. El ranking inducido es coherente, aunque las preferencias humanas reales pueden violarlo.

---

## 3. Derivacion: por que sigmoide

¿Por que la sigmoide y no otra funcion creciente acotada en $[0, 1]$? Tres caminos llevan al mismo destino:

### 3.1 Maximum entropy

Si solo conocemos los valores esperados $E[r_w - r_l]$ bajo la distribucion de preferencias, el principio de maxima entropia da la familia exponencial -- y la binaria con feature $r_w - r_l$ es exactamente la logistica.

### 3.2 Modelo de utilidad ruidosa

Asumamos que cada item tiene una utilidad latente $U_i = r_i + \epsilon_i$ donde $\epsilon_i$ sigue una distribucion **Gumbel** (extreme value tipo I). El item ganador es el de mayor $U$. Entonces:

$$
P(y_w \succeq y_l) = P(U_w > U_l) = P(\epsilon_l - \epsilon_w < r_w - r_l)
$$

La diferencia de dos Gumbels independientes es **logistica**, asi que esta probabilidad es exactamente $\sigma(r_w - r_l)$. Este truco -- conocido como Gumbel-max -- aparece tambien en sampling categorico.

### 3.3 Coincidencia con regresion logistica binaria

Si codificamos la preferencia como variable binaria $z = 1$ si $y_w$ gana y $z = 0$ si pierde, y usamos la diferencia $\Delta = r_w - r_l$ como feature, entonces Bradley-Terry **es** regresion logistica con peso 1 y sin sesgo:

$$
P(z=1 \mid \Delta) = \sigma(\Delta).
$$

Es la misma maquinaria de optimizacion (cross-entropy convex, sin minimos locales).

---

## 4. Maximum likelihood sobre dataset de preferencias

Con dataset $\mathcal{D} = \{(x_i, y_w^i, y_l^i)\}_{i=1}^N$ -- donde $x_i$ es el contexto (prompt) y $y_w^i \succ y_l^i$ es el juicio pareado -- la log-likelihood bajo Bradley-Terry es:

$$
\log \mathcal{L}(r) = \sum_{i=1}^N \log \sigma(r(x_i, y_w^i) - r(x_i, y_l^i)).
$$

Maximizar esta cantidad equivale a minimizar la **loss Bradley-Terry**:

$$
\mathcal{L}_{\text{BT}}(r) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \big[ \log \sigma(r(x, y_w) - r(x, y_l)) \big].
$$

Si $r$ es lineal en features -- $r(x, y) = \theta^T \phi(x, y)$ -- la optimizacion es **convexa** y se resuelve con descenso de gradiente o IRLS estandar. Si $r$ es una red neuronal (un MLP encima de los embeddings de un Transformer, como en RLHF clasico), la convexidad se pierde pero el setup sigue funcionando empiricamente bien.

Identificabilidad: como solo importa la diferencia, debemos fijar una constante (ej. $\sum_i r_i = 0$ o $r_{y_0} = 0$ para algun item de referencia). Sin esa restriccion, el optimo es una linea en el espacio de parametros.

---

## 5. Conexion con Plackett-Luce: rankings n-way

Bradley-Terry maneja **pares**. ¿Que pasa si el juicio humano es un **ranking** completo de $K$ items, no solo un par? La generalizacion natural es **Plackett-Luce** (Plackett 1975, Luce 1959):

$$
P(\sigma) = \prod_{k=1}^K \frac{e^{r_{\sigma(k)}}}{\sum_{j=k}^K e^{r_{\sigma(j)}}}.
$$

Es decir, la probabilidad de un ranking $\sigma$ es el producto de "elegir el primero de los restantes" en cada paso. Plackett-Luce se reduce a Bradley-Terry para $K=2$.

En la practica de RLHF/DPO, casi todos los datasets son pares ($K=2$), porque pedir rankings completos a anotadores es caro y mas ruidoso. Pero hay variantes (LiPO, Plackett-Luce-DPO) que extienden la idea.

---

## 6. Uso en RLHF clasico: el reward model

En la receta original de RLHF (Christiano 2017, Ouyang 2022 / InstructGPT), Bradley-Terry justifica el **reward model**:

1. Coleccionas dataset $\mathcal{D}$ de preferencias humanas $(x, y_w, y_l)$.
2. Entrenas un MLP $r_\phi(x, y)$ encima del Transformer SFT con la loss Bradley-Terry sobre $\mathcal{D}$.
3. Una vez convergido $r_\phi$, lo usas como **funcion de reward** para PPO: la policy $\pi_\theta$ se optimiza para maximizar $\mathbb{E}[r_\phi(x, y)]$ con un termino de [KL implicito](/fundamentos/kl-implicito) hacia el SFT.

El paso (2) es Bradley-Terry puro. El paso (3) es donde DPO va a hacer la magia: en vez de aprender $r_\phi$ explicitamente y despues hacer PPO, DPO sustituye $r_\phi$ por una expresion **algebraica** en terminos de la policy y la referencia.

---

## 7. Uso en DPO: forma implicita

La revolucion conceptual de DPO (Rafailov 2023): en el optimo de RLHF con KL-regularizacion contra $\pi_{\text{ref}}$, existe una **forma cerrada** de la policy optima:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right).
$$

Invirtiendo:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

El termino $\beta \log Z(x)$ depende solo de $x$, asi que **se cancela** cuando aparece en una **diferencia** $r(x, y_w) - r(x, y_l)$. Sustituyendo en la loss Bradley-Terry obtenemos:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right).
$$

Es Bradley-Terry, exactamente, con el reward $r$ reemplazado por $\beta \log \pi_\theta / \pi_{\text{ref}}$. El log-ratio se llama **reward implicito** y la consecuencia es una loss que opera **directamente sobre la policy** -- sin reward model separado, sin PPO.

{{< concept-alert type="clave" >}}
**DPO no es una alternativa a Bradley-Terry**. Es Bradley-Terry con un reward parametrizado por log-ratios de policy. La novedad esta en la parametrizacion, no en la loss de preferencias.
{{< /concept-alert >}}

---

## 8. Resumen

- **Bradley-Terry** modela la probabilidad de preferencia entre dos opciones como $\sigma(r_w - r_l)$, donde $r$ es la fortaleza latente de cada opcion.
- Solo importa la **diferencia** de fortalezas; el modelo es invariante a translaciones.
- Tres derivaciones llevan a la sigmoide: max entropy, utilidad ruidosa Gumbel, y regresion logistica binaria.
- Maximum likelihood sobre dataset de pares da la **loss Bradley-Terry**, $-\mathbb{E}\log\sigma(r_w - r_l)$, convexa en $r$ lineal.
- **Plackett-Luce** generaliza a rankings de $K$ items; Bradley-Terry es el caso $K=2$.
- En **RLHF clasico**, justifica el reward model entrenado a partir de preferencias humanas.
- En **DPO**, la misma loss reaparece con $r$ sustituido por $\beta \log \pi_\theta/\pi_{\text{ref}}$ -- el reward implicito.

## Ver tambien

- [DPO](/fundamentos/dpo) -- usa la loss Bradley-Terry con reward implicito.
- [SFT](/fundamentos/sft) -- el paso previo que produce el $\pi_{\text{ref}}$.
- [KL Implicito](/fundamentos/kl-implicito) -- el regularizador que justifica la forma cerrada de la policy optima.
- [Funciones de Perdida](/fundamentos/funciones-perdida) -- contexto general sobre cross-entropy y losses convexas.
- [Clase 14 cap 26 - Preferencias y Bradley-Terry](/clases/clase-14/practica/26-preferencias-bradley-terry) -- desarrollo en el curso.
- [Clase 14 cap 27 - DPO loss](/clases/clase-14/practica/27-dpo-loss) -- la derivacion paso a paso.
