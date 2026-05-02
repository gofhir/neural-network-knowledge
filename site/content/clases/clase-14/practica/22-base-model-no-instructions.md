---
title: "22 - El base model no sigue instrucciones"
weight: 220
math: true
---

Termino el Camino 1 con un Mini-LLaMA que predice caracteres al estilo Shakespeare. Es un modelo base: dado un contexto, devuelve el siguiente caracter mas probable. Eso es todo lo que sabe hacer. Ahora queremos algo distinto — que **siga instrucciones**. Antes de cambiar nada, conviene probar lo que ya tenemos y ver concretamente que falla. Este capitulo abre el **Camino 2** (capitulos 22-29), que cubre **SFT** (caps 23-25) y **DPO** (caps 26-29). Para repasar el modelo base que vamos a poner a prueba, [cap 21](../21-mini-llama).

---

## 1. La pregunta motivadora

Tenemos un modelo entrenado en Shakespeare que predice el siguiente caracter. Si le damos un prompt con formato `INSTR: ... \nRESP: `, ¿que pasa?

La respuesta intuitiva es: nada bueno. Pero conviene verlo con los ojos. La intuicion sin output literal se vuelve handwaving — y el resto del Camino 2 depende de entender, en el cuerpo, **por que** el modelo no responde.

---

## 2. El experimento

Cargamos el Mini-LLaMA pretrained desde el checkpoint de Camino 1, le damos cuatro prompts en formato instruccion/respuesta (tres con `INSTR/RESP`, uno con `Q/A`), y simplemente miramos lo que genera. Sin trampas: misma seed, misma temperatura, mismo top-k que cualquier generacion del modelo base.

```python
"""14_show_base_no_instructions.py - Cap 22: el problema.

El Mini-LLaMA pretrained ignora el formato INSTR/RESP y genera Shakespeare-ish.
Este script lo demuestra dandole prompts de instruccion y mostrando el output.
"""
import torch
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps

torch.manual_seed(1337)

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt")

prompts = [
    "INSTR: reverse 'cat'\nRESP: ",
    "INSTR: upper 'hello'\nRESP: ",
    "INSTR: repeat 'a' 3\nRESP: ",
    "Q: who wrote Hamlet?\nA: ",
]

print("=== Mini-LLaMA base (Camino 1) frente a prompts de instruccion ===\n")
for p in prompts:
    print(f"--- Prompt ---\n{p}")
    print(f"--- Output ---")
    out = generate_with_prompt(model, p, c2i, i2c, max_new_tokens=40,
                               temperature=0.8, top_k=10)
    print(out)
    print()
```

Veinticinco lineas. Cargar checkpoint, definir prompts, generar, imprimir. Ningun truco.

---

## 3. El output literal

Esto es lo que imprime el script al correrlo:

```
=== Mini-LLaMA base (Camino 1) frente a prompts de instrucción ===

--- Prompt ---
INSTR: reverse 'cat'
RESP: 
--- Output ---
INSTR: reverse 'cat'
RESP: alast the king, there is be doth in him.

--- Prompt ---
INSTR: upper 'hello'
RESP: 
--- Output ---
INSTR: upper 'hello'
RESP: say, that it? that this bright,


--- Prompt ---
INSTR: repeat 'a' 3
RESP: 
--- Output ---
INSTR: repeat 'a' 3
RESP: what I do be never soon and subjects man

--- Prompt ---
Q: who wrote Hamlet?
A: 
--- Output ---
Q: who wrote Hamlet?
A: he is you have be servicious both,
```

---

## 4. Analisis

**Lo que vemos.** En los cuatro casos, el modelo **copia el prompt** (porque la generacion empieza desde el prompt y va concatenando tokens nuevos) y luego, donde deberia estar la respuesta, sigue generando texto Shakespeare-ish. "alast the king, there is be doth in him". "say, that it? that this bright". "what I do be never soon and subjects man". "he is you have be servicious both". Vocabulario isabelino, sintaxis aproximadamente coherente, **cero respeto al formato `INSTR/RESP`**. El modelo no invierte "cat", no pasa "hello" a mayusculas, no repite "a" tres veces, no sabe que Shakespeare escribio Hamlet.

**Por que pasa.** El Mini-LLaMA solo vio Shakespeare durante pretraining. Para el, `INSTR:` es un prefijo que aparece muy raro o nunca en el corpus. El modelo no tiene noción de "instruccion" ni de "respuesta" — solo continua la distribucion del corpus. Cuando ve `INSTR: reverse 'cat'\nRESP: `, lo trata como un fragmento raro de texto y genera la continuacion mas plausible **dada la distribucion de Shakespeare**. Y la distribucion de Shakespeare no incluye respuestas a instrucciones — incluye reyes, soldados, "doth", "thou", monologos. Por eso aparecen "the king", "doth", "subjects".

**Lo que falta.** Necesitamos enseñarle el formato. La tecnica es **fine-tuning supervisado** (SFT, Supervised Fine-Tuning): le mostramos pares `(INSTR ..., RESP correcta)` y lo entrenamos a generar la respuesta dada el prompt. No es magia — es seguir entrenando el mismo modelo, con el mismo loss de cross-entropy, pero sobre un dataset distinto: pares prompt/respuesta en vez de Shakespeare crudo. Eso es lo que viene en los caps 23-25.

**Conexion con DPO.** SFT ensena el formato — despues de SFT el modelo sabe que un `INSTR:` se sigue de una `RESP:` y que esa `RESP:` debe estar relacionada con la instruccion. Pero SFT no le ensena **preferencias**: dado dos respuestas posibles, ¿cual es mejor? Eso lo hace **DPO** (Direct Preference Optimization, caps 26-29), que ajusta el modelo con pares `(respuesta preferida, respuesta rechazada)`. Ambos son necesarios: SFT da el formato, DPO refina el comportamiento.

---

## 5. Preguntas de verificacion

1. ¿Por que el modelo no respeta el formato `INSTR/RESP`?
2. Si bajamos `temperature` a 0.2, ¿el modelo seguiria instrucciones? ¿Por que no?
3. ¿Que tendria que cambiar en el dataset de pretraining para que el modelo siguiera instrucciones sin SFT?

Pista para la 1: el modelo solo modela la distribucion de los datos que vio.

Pista para la 2: temperature solo afecta la varianza del sampling sobre la distribucion que el modelo ya aprendio. No le ensena patrones nuevos. Bajar `temperature` haria que el output sea **mas Shakespeare-ish**, no menos.

Pista para la 3: el dataset tendria que incluir ejemplos del formato `INSTR/RESP` (o `Q/A`). Es decir, tendria que ser un dataset mixto pretraining + instrucciones — o directamente, hacer SFT despues. Empiricamente, lo segundo funciona mejor: pretraining masivo en texto crudo + SFT focalizado en instrucciones.

---

## 6. Que viene

En el [cap 23](../23-dataset-sft) construimos un dataset sintetico de 4 tareas (reverse, upper, repeat, length) para SFT. Despues de SFT (caps 24-25), el mismo modelo va a responder estos cuatro prompts correctamente — sin cambiar la arquitectura, solo cambiando el dataset y siguiendo entrenando.

Volver al [hub de practica](..) o a la [Clase 14](../..).
