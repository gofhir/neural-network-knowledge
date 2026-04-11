# Diseño: MLP de Plausibilidad Clínica UCUM

**Fecha:** 2026-04-05  
**Proyecto:** fhir-plausibility  
**Objetivo:** Entrenar un MLP en PyTorch que detecte valores clínicos implausibles (errores de unidad, outliers), exportarlo a ONNX, y consumirlo desde el FHIR server en Go.

---

## Contexto

El UCUM engine en Go valida que una unidad sea dimensionalmente correcta para un tipo de observación (Capa 1). Este modelo cubre la Capa 2: dado que la unidad es válida, ¿es el valor numérico fisiológicamente plausible?

**Ejemplos de errores que detecta:**
- Hemoglobina 150 g/dL (debería ser g/L — error de conversión × 10)
- Hemoglobina 0.8 g/dL (división por 18 — confusión de sistema de unidades)
- Presión sistólica 1200 mmHg (outlier estadístico imposible)

---

## Entorno

- **Hardware:** Mac M4 (Apple Silicon)
- **GPU:** MPS (Metal Performance Shaders) vía PyTorch
- **Gestor de entorno:** `uv`
- **Python:** 3.11+

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init fhir-plausibility
cd fhir-plausibility
uv add torch torchvision
uv add onnx onnxruntime scikit-learn pandas numpy matplotlib jupyter
```

---

## Estructura del Proyecto

```
fhir-plausibility/
├── data/
│   ├── synthea/          ← FHIR JSON generado por Synthea
│   └── mimic/            ← MIMIC-IV descargado de PhysioNet
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_entrenamiento.ipynb
│   └── 03_export_onnx.ipynb
├── src/
│   ├── dataset.py        ← carga, parsea y genera negativos
│   ├── model.py          ← arquitectura MLP
│   ├── train.py          ← loop de entrenamiento con MPS
│   └── export.py         ← exporta a ONNX
├── models/
│   └── plausibility.onnx ← modelo final
└── pyproject.toml
```

---

## Arquitectura del Modelo

```
Input: [valor_normalizado, loinc_code_id, ucum_unit_id, edad_normalizada, sexo_onehot]
          ↓
    Linear(input_dim → 64) + ReLU + Dropout(0.2)
          ↓
    Linear(64 → 32) + ReLU + Dropout(0.2)
          ↓
    Linear(32 → 1) + Sigmoid
          ↓
Output: probabilidad de plausibilidad [0.0 - 1.0]
```

**Loss:** BCELoss (clasificación binaria)  
**Optimizer:** Adam (lr=1e-3)  
**Batch size:** 256  
**Epochs:** 50  
**Device:** MPS (M4) con fallback a CPU

---

## Pipeline de Datos

### Fuentes
- **Synthea:** datos FHIR sintéticos (sin restricciones, fácil de generar)
- **MIMIC-IV Demo:** datos clínicos reales (PhysioNet, ya registrado)

### Generación de Etiquetas

**Positivos (label=1):** observaciones reales del dataset  
**Negativos (label=0):** generados automáticamente con errores realistas:

```python
# 1. Error de conversión (factor fisiológico conocido)
#    g/dL ↔ mmol/L para hemoglobina: factor = 0.6206
valor_negativo = valor_real * 18     # o / 18

# 2. Outlier estadístico (> 3 desviaciones estándar)
valor_negativo = media + 5 * std

# 3. Valor de otra observación (contexto incorrecto)
# asignar valor de glucosa al campo de hemoglobina
```

**Proporción:** 70% positivos / 30% negativos  
**División:** 80% train / 10% val / 10% test

### Features de Entrada

| Feature | Tipo | Encoding |
|---|---|---|
| valor numérico | float | normalización z-score por (LOINC, unidad) |
| loinc_code | categórico | embedding entero (lookup table) |
| ucum_unit | categórico | embedding entero (lookup table) |
| edad | float | normalización min-max |
| sexo | binario | one-hot [0, 1] |

---

## Entrenamiento

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = PlausibilityNet(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
```

**Métricas a monitorear:**
- Loss (train + val)
- Accuracy
- Precisión / Recall / F1 (importante: los falsos negativos son costosos)
- AUC-ROC

---

## Export a ONNX

```python
model.eval().to("cpu")
dummy_input = torch.randn(1, input_dim)

torch.onnx.export(
    model, dummy_input,
    "models/plausibility.onnx",
    input_names=["features"],
    output_names=["plausibility_score"],
    opset_version=17
)
```

**Verificación:** cargar con `onnxruntime` en Python antes de pasar a Go.

---

## Integración con Go FHIR Server

```
FHIR write path
    → UCUM engine (validación dimensional — Capa 1)
    → feature extractor (Go)
    → onnxruntime_go (inferencia ONNX — Capa 2, < 2ms)
    → score < threshold: DetectedIssue resource + AuditEvent
    → score >= threshold: continuar pipeline normal
```

---

## Criterios de Éxito

- Accuracy > 90% en test set
- F1 score > 0.88
- Inferencia < 5ms en CPU (para Go)
- Modelo exportado y verificado con `onnxruntime`
- Integración demostrable con un recurso FHIR de ejemplo

---

## Próximos Pasos (después de este proyecto)

**Proyecto B:** Siamese Network para Master Patient Index (MPI)  
- Aprende similitud entre pares de pacientes
- Contrastive loss / Triplet loss
- Output: `Patient.link` automático en el FHIR server
