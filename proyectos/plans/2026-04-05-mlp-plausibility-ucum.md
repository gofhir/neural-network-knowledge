# MLP Plausibilidad Clínica UCUM — Plan de Implementación

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Entrenar un MLP en PyTorch que detecte valores clínicos implausibles, exportarlo a ONNX, y verificar que corre desde Python antes de integrarlo en Go.

**Architecture:** Pipeline de datos desde Synthea + MIMIC-IV → features tabulares → MLP binario (plausible/no plausible) entrenado con MPS en M4 → export ONNX → verificación con onnxruntime.

**Tech Stack:** Python 3.11, uv, PyTorch (MPS), pandas, scikit-learn, onnx, onnxruntime, Jupyter

---

## Task 1: Setup del entorno con uv

**Files:**
- Create: `~/projects/fhir-plausibility/pyproject.toml` (generado por uv)

**Step 1: Instalar uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reiniciar terminal o ejecutar:
source ~/.zshrc
```

Expected: `uv 0.x.x` al correr `uv --version`

**Step 2: Crear el proyecto**

```bash
cd ~/projects
uv init fhir-plausibility
cd fhir-plausibility
```

Expected: carpeta `fhir-plausibility/` con `pyproject.toml` y `hello.py`

**Step 3: Agregar dependencias**

```bash
uv add torch torchvision
uv add onnx onnxruntime
uv add scikit-learn pandas numpy matplotlib
uv add jupyter ipykernel
```

Expected: `pyproject.toml` actualizado, entorno virtual en `.venv/`

**Step 4: Verificar PyTorch con MPS**

```bash
uv run python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS disponible:', torch.backends.mps.is_available())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
x = torch.randn(3, 3).to(device)
print('Tensor en', device, ':', x.shape)
"
```

Expected:
```
PyTorch: 2.x.x
MPS disponible: True
Tensor en mps : torch.Size([3, 3])
```

**Step 5: Crear estructura de carpetas**

```bash
mkdir -p data/synthea data/mimic notebooks src models
touch src/__init__.py
```

**Step 6: Commit inicial**

```bash
git init
git add pyproject.toml uv.lock
git commit -m "feat: setup inicial con uv y PyTorch MPS"
```

---

## Task 2: Generar datos con Synthea

**Files:**
- Create: `data/synthea/` (poblado por Synthea)
- Create: `src/synthea_parser.py`

**Step 1: Descargar y correr Synthea**

```bash
# Opción A: con Java instalado
curl -LO https://github.com/synthetichealth/synthea/releases/latest/download/synthea-with-dependencies.jar
java -jar synthea-with-dependencies.jar -p 1000 --exporter.fhir.export=true
# Los archivos FHIR JSON quedan en output/fhir/

# Mover a nuestro proyecto
mv output/fhir/*.json data/synthea/
```

```bash
# Opción B: si no tienes Java, descargar dataset pre-generado
curl -LO https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip
unzip synthea_sample_data_fhir_r4_sep2019.zip -d data/synthea/
```

Expected: archivos `*.json` en `data/synthea/`

**Step 2: Crear parser de Synthea**

Crear `src/synthea_parser.py`:

```python
import json
import os
import pandas as pd
from pathlib import Path


# LOINCs de observaciones clínicas comunes con sus rangos fisiológicos
LOINC_RANGES = {
    "718-7":  {"name": "hemoglobina",         "unit": "g/dL",  "mean": 13.5, "std": 2.0,  "min": 4.0,  "max": 22.0},
    "2339-0": {"name": "glucosa",             "unit": "mg/dL", "mean": 100,  "std": 25,   "min": 40,   "max": 500},
    "55284-4":{"name": "presion_arterial_s",  "unit": "mm[Hg]","mean": 120,  "std": 20,   "min": 60,   "max": 250},
    "8867-4": {"name": "frecuencia_cardiaca", "unit": "/min",  "mean": 75,   "std": 15,   "min": 30,   "max": 200},
    "2160-0": {"name": "creatinina",          "unit": "mg/dL", "mean": 1.0,  "std": 0.3,  "min": 0.3,  "max": 15.0},
}


def parse_synthea_bundle(filepath: str) -> list[dict]:
    """Extrae observaciones relevantes de un Bundle FHIR de Synthea."""
    with open(filepath) as f:
        bundle = json.load(f)

    observations = []
    patient_age = None
    patient_sex = None

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})

        # Extraer datos del paciente
        if resource.get("resourceType") == "Patient":
            patient_sex = 1 if resource.get("gender") == "male" else 0
            birth_year = int(resource.get("birthDate", "1980")[:4])
            patient_age = 2024 - birth_year

        # Extraer observaciones
        if resource.get("resourceType") == "Observation":
            coding = resource.get("code", {}).get("coding", [{}])[0]
            loinc = coding.get("code", "")

            if loinc not in LOINC_RANGES:
                continue

            value_quantity = resource.get("valueQuantity", {})
            value = value_quantity.get("value")
            unit = value_quantity.get("code", "")

            if value is None:
                continue

            observations.append({
                "loinc_code": loinc,
                "value": float(value),
                "ucum_unit": unit,
                "age": patient_age or 40,
                "sex": patient_sex if patient_sex is not None else 0,
                "label": 1,  # todos los reales son positivos
                "source": "synthea",
            })

    return observations


def load_synthea(data_dir: str) -> pd.DataFrame:
    """Carga todos los bundles Synthea de un directorio."""
    all_obs = []
    path = Path(data_dir)

    for fpath in path.glob("*.json"):
        try:
            obs = parse_synthea_bundle(str(fpath))
            all_obs.extend(obs)
        except Exception as e:
            print(f"Error en {fpath.name}: {e}")

    df = pd.DataFrame(all_obs)
    print(f"Synthea: {len(df)} observaciones cargadas")
    return df


if __name__ == "__main__":
    df = load_synthea("data/synthea")
    print(df.head())
    print(df["loinc_code"].value_counts())
```

**Step 3: Verificar parser**

```bash
uv run python src/synthea_parser.py
```

Expected:
```
Synthea: XXXX observaciones cargadas
loinc_code
718-7     XXX
2339-0    XXX
...
```

**Step 4: Commit**

```bash
git add src/synthea_parser.py
git commit -m "feat: parser de observaciones FHIR desde Synthea"
```

---

## Task 3: Cargar datos MIMIC-IV

**Files:**
- Create: `src/mimic_parser.py`

**Prerequisito:** Descargar desde PhysioNet (ya registrado):
- Ir a https://physionet.org/content/mimic-iv-fhir-demo/
- Descargar `mimic-iv-fhir-demo.zip`
- Extraer en `data/mimic/`

**Step 1: Crear parser de MIMIC-IV**

Crear `src/mimic_parser.py`:

```python
import json
import pandas as pd
from pathlib import Path
from synthea_parser import LOINC_RANGES


def parse_mimic_observation(resource: dict, age: int = 60, sex: int = 0) -> dict | None:
    """Parsea un recurso Observation de MIMIC-IV FHIR."""
    coding = resource.get("code", {}).get("coding", [{}])[0]
    loinc = coding.get("code", "")

    if loinc not in LOINC_RANGES:
        return None

    value_quantity = resource.get("valueQuantity", {})
    value = value_quantity.get("value")
    unit = value_quantity.get("code", "")

    if value is None:
        return None

    return {
        "loinc_code": loinc,
        "value": float(value),
        "ucum_unit": unit,
        "age": age,
        "sex": sex,
        "label": 1,
        "source": "mimic",
    }


def load_mimic(data_dir: str) -> pd.DataFrame:
    """Carga observaciones de MIMIC-IV FHIR."""
    all_obs = []
    path = Path(data_dir)

    for fpath in path.rglob("Observation*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)

            # Puede ser un Bundle o un recurso individual
            if data.get("resourceType") == "Bundle":
                for entry in data.get("entry", []):
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") == "Observation":
                        obs = parse_mimic_observation(resource)
                        if obs:
                            all_obs.append(obs)
            elif data.get("resourceType") == "Observation":
                obs = parse_mimic_observation(data)
                if obs:
                    all_obs.append(obs)

        except Exception as e:
            print(f"Error en {fpath.name}: {e}")

    df = pd.DataFrame(all_obs)
    print(f"MIMIC-IV: {len(df)} observaciones cargadas")
    return df


if __name__ == "__main__":
    df = load_mimic("data/mimic")
    print(df.head())
```

**Step 2: Verificar parser**

```bash
uv run python src/mimic_parser.py
```

Expected: `MIMIC-IV: XXXX observaciones cargadas`

**Step 3: Commit**

```bash
git add src/mimic_parser.py
git commit -m "feat: parser de observaciones FHIR desde MIMIC-IV"
```

---

## Task 4: Construir Dataset con Negativos Realistas

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Escribir tests primero (TDD)**

Crear `tests/test_dataset.py`:

```python
import pandas as pd
import pytest
from src.dataset import generate_negatives, build_dataset, PlausibilityDataset
import torch


def make_sample_df():
    return pd.DataFrame([
        {"loinc_code": "718-7", "value": 14.5, "ucum_unit": "g/dL", "age": 45, "sex": 1, "label": 1, "source": "test"},
        {"loinc_code": "718-7", "value": 13.0, "ucum_unit": "g/dL", "age": 30, "sex": 0, "label": 1, "source": "test"},
        {"loinc_code": "2339-0","value": 95.0, "ucum_unit": "mg/dL","age": 55, "sex": 1, "label": 1, "source": "test"},
    ])


def test_generate_negatives_returns_dataframe():
    df = make_sample_df()
    neg = generate_negatives(df)
    assert isinstance(neg, pd.DataFrame)


def test_generate_negatives_all_labeled_zero():
    df = make_sample_df()
    neg = generate_negatives(df)
    assert (neg["label"] == 0).all()


def test_generate_negatives_same_loinc():
    df = make_sample_df()
    neg = generate_negatives(df)
    # Negativos deben mantener el mismo loinc_code
    assert set(neg["loinc_code"]).issubset(set(df["loinc_code"]))


def test_build_dataset_has_both_labels():
    df = make_sample_df()
    full = build_dataset(df)
    assert 0 in full["label"].values
    assert 1 in full["label"].values


def test_pytorch_dataset_returns_tensors():
    df = make_sample_df()
    full = build_dataset(df)
    ds = PlausibilityDataset(full)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
```

**Step 2: Correr tests para verificar que fallan**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: `FAILED` — `src/dataset.py` no existe aún

**Step 3: Implementar `src/dataset.py`**

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from synthea_parser import LOINC_RANGES


# Factores de conversión típicos que causan errores clínicos reales
CONVERSION_ERRORS = {
    "718-7":   [18.0, 0.0621],    # hemoglobina: g/dL ↔ mmol/L
    "2339-0":  [0.0555, 18.0],    # glucosa: mg/dL ↔ mmol/L
    "2160-0":  [88.4, 0.0113],    # creatinina: mg/dL ↔ umol/L
}


def generate_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Genera ejemplos negativos (implausibles) a partir de observaciones reales."""
    negatives = []

    for _, row in df.iterrows():
        loinc = row["loinc_code"]
        value = row["value"]
        info = LOINC_RANGES.get(loinc, {})

        # Estrategia 1: error de conversión (el más frecuente en clínica)
        if loinc in CONVERSION_ERRORS:
            for factor in CONVERSION_ERRORS[loinc]:
                neg_value = value * factor
                # Solo agregar si el valor resultante es realmente implausible
                if neg_value < info.get("min", 0) * 0.5 or neg_value > info.get("max", 9999) * 2:
                    neg_row = row.copy()
                    neg_row["value"] = neg_value
                    neg_row["label"] = 0
                    negatives.append(neg_row)

        # Estrategia 2: outlier estadístico (> 4 desviaciones estándar)
        mean = info.get("mean", value)
        std = info.get("std", 1)
        for direction in [1, -1]:
            outlier_value = mean + direction * 5 * std
            if outlier_value > 0:  # valores negativos no tienen sentido
                neg_row = row.copy()
                neg_row["value"] = outlier_value
                neg_row["label"] = 0
                negatives.append(neg_row)

    return pd.DataFrame(negatives)


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Combina positivos y negativos en un dataset balanceado."""
    positives = df[df["label"] == 1].copy()
    negatives = generate_negatives(positives)

    # Limitar negativos al 30% del total
    n_neg = min(len(negatives), int(len(positives) * 0.43))  # 70/30 ratio
    negatives = negatives.sample(n=n_neg, random_state=42)

    full = pd.concat([positives, negatives], ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"Dataset: {len(positives)} positivos + {len(negatives)} negativos = {len(full)} total")
    return full


def encode_features(df: pd.DataFrame):
    """Codifica features categóricas y normaliza numéricas."""
    loinc_enc = LabelEncoder()
    unit_enc = LabelEncoder()

    df = df.copy()
    df["loinc_id"] = loinc_enc.fit_transform(df["loinc_code"])
    df["unit_id"] = unit_enc.fit_transform(df["ucum_unit"])

    # Normalizar valor por (loinc_code) usando z-score
    df["value_norm"] = df.groupby("loinc_code")["value"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df["age_norm"] = (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-8)

    return df, loinc_enc, unit_enc


class PlausibilityDataset(Dataset):
    """Dataset PyTorch para el modelo de plausibilidad."""

    def __init__(self, df: pd.DataFrame):
        df, _, _ = encode_features(df)

        features = df[["value_norm", "loinc_id", "unit_id", "age_norm", "sex"]].values
        labels = df["label"].values

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

**Step 4: Correr tests y verificar que pasan**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: todos en `PASSED`

**Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: dataset con generación de negativos realistas (TDD)"
```

---

## Task 5: Arquitectura del Modelo MLP

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

**Step 1: Escribir tests del modelo**

Crear `tests/test_model.py`:

```python
import torch
from src.model import PlausibilityNet


def test_model_output_shape():
    model = PlausibilityNet(input_dim=5)
    x = torch.randn(32, 5)  # batch de 32
    out = model(x)
    assert out.shape == (32, 1)


def test_model_output_between_0_and_1():
    model = PlausibilityNet(input_dim=5)
    x = torch.randn(100, 5)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_model_single_sample():
    model = PlausibilityNet(input_dim=5)
    x = torch.randn(1, 5)
    out = model(x)
    assert out.shape == (1, 1)
```

**Step 2: Correr tests para verificar que fallan**

```bash
uv run pytest tests/test_model.py -v
```

Expected: `FAILED`

**Step 3: Implementar `src/model.py`**

```python
import torch
import torch.nn as nn


class PlausibilityNet(nn.Module):
    """
    MLP binario para detectar valores clínicos implausibles.
    
    Input:  [valor_norm, loinc_id, unit_id, age_norm, sex]
    Output: probabilidad de plausibilidad [0.0 - 1.0]
    """

    def __init__(self, input_dim: int = 5, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**Step 4: Correr tests y verificar que pasan**

```bash
uv run pytest tests/test_model.py -v
```

Expected: todos en `PASSED`

**Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: arquitectura MLP PlausibilityNet (TDD)"
```

---

## Task 6: Loop de Entrenamiento con MPS

**Files:**
- Create: `src/train.py`

**Step 1: Implementar `src/train.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from pathlib import Path

from dataset import PlausibilityDataset, build_dataset
from model import PlausibilityNet
from synthea_parser import load_synthea


def train(data_dir: str = "data/synthea", epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
    # Detectar dispositivo
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Entrenando en: {device}")

    # Cargar datos
    df_synthea = load_synthea(data_dir)

    # Si existe MIMIC-IV, agregar
    mimic_path = Path("data/mimic")
    if mimic_path.exists() and any(mimic_path.rglob("*.json")):
        from mimic_parser import load_mimic
        df_mimic = load_mimic(str(mimic_path))
        df = pd.concat([df_synthea, df_mimic], ignore_index=True)
    else:
        df = df_synthea

    # Construir dataset con negativos
    full_df = build_dataset(df)
    dataset = PlausibilityDataset(full_df)

    # Split 80/10/10
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # Modelo, loss, optimizer
    model = PlausibilityNet(input_dim=5).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loop de entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validación
        if epoch % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = (model(X_batch).squeeze() > 0.5).float()
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)
            val_acc = correct / total * 100
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1f}%")

    # Evaluación final en test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = (model(X_batch).squeeze() > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    test_acc = correct / total * 100
    print(f"\nTest Accuracy: {test_acc:.1f}%")

    # Guardar modelo
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/plausibility.pt")
    print("Modelo guardado en models/plausibility.pt")

    return model


if __name__ == "__main__":
    train()
```

**Step 2: Correr entrenamiento**

```bash
cd src
uv run python train.py
```

Expected (ejemplo):
```
Entrenando en: mps
Synthea: 5432 observaciones cargadas
Dataset: 5432 positivos + 2328 negativos = 7760 total
Epoch  10/50 | Loss: 0.3124 | Val Acc: 87.4%
Epoch  20/50 | Loss: 0.2418 | Val Acc: 91.2%
Epoch  30/50 | Loss: 0.1987 | Val Acc: 93.1%
Epoch  40/50 | Loss: 0.1654 | Val Acc: 94.5%
Epoch  50/50 | Loss: 0.1423 | Val Acc: 95.2%

Test Accuracy: 94.8%
Modelo guardado en models/plausibility.pt
```

**Step 3: Commit**

```bash
cd ..
git add src/train.py models/
git commit -m "feat: loop de entrenamiento con MPS y evaluación"
```

---

## Task 7: Export a ONNX

**Files:**
- Create: `src/export.py`
- Create: `tests/test_export.py`

**Step 1: Escribir test de export**

Crear `tests/test_export.py`:

```python
import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from src.model import PlausibilityNet


def test_onnx_export_creates_file(tmp_path):
    model = PlausibilityNet(input_dim=5)
    out_path = str(tmp_path / "test_model.onnx")

    dummy = torch.randn(1, 5)
    torch.onnx.export(model.to("cpu"), dummy, out_path,
                      input_names=["features"], output_names=["score"],
                      opset_version=17)

    assert Path(out_path).exists()


def test_onnx_inference_matches_pytorch(tmp_path):
    model = PlausibilityNet(input_dim=5)
    model.eval()
    out_path = str(tmp_path / "test_model.onnx")

    dummy = torch.randn(1, 5)
    torch.onnx.export(model.to("cpu"), dummy, out_path,
                      input_names=["features"], output_names=["score"],
                      opset_version=17)

    # Inferencia PyTorch
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # Inferencia ONNX
    sess = ort.InferenceSession(out_path)
    onnx_out = sess.run(["score"], {"features": dummy.numpy()})[0]

    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4)


def test_onnx_output_between_0_and_1(tmp_path):
    model = PlausibilityNet(input_dim=5)
    out_path = str(tmp_path / "test_model.onnx")

    dummy = torch.randn(1, 5)
    torch.onnx.export(model.to("cpu"), dummy, out_path,
                      input_names=["features"], output_names=["score"],
                      opset_version=17)

    sess = ort.InferenceSession(out_path)
    for _ in range(10):
        x = np.random.randn(1, 5).astype(np.float32)
        out = sess.run(["score"], {"features": x})[0]
        assert 0.0 <= out[0][0] <= 1.0
```

**Step 2: Correr tests para verificar que fallan**

```bash
uv run pytest tests/test_export.py -v
```

Expected: `FAILED` — modelo no exportado aún

**Step 3: Implementar `src/export.py`**

```python
import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from model import PlausibilityNet


def export_to_onnx(
    weights_path: str = "models/plausibility.pt",
    output_path: str = "models/plausibility.onnx",
    input_dim: int = 5,
):
    """Exporta el modelo entrenado a formato ONNX."""
    # Cargar modelo entrenado
    model = PlausibilityNet(input_dim=input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["features"],
        output_names=["plausibility_score"],
        dynamic_axes={"features": {0: "batch_size"}},  # batch variable
        opset_version=17,
    )
    print(f"Modelo exportado a {output_path}")

    # Verificar con onnxruntime
    sess = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    result = sess.run(["plausibility_score"], {"features": test_input})[0]
    print(f"Verificación ONNX OK — score de prueba: {result[0][0]:.4f}")

    return output_path


if __name__ == "__main__":
    export_to_onnx()
```

**Step 4: Exportar el modelo entrenado**

```bash
cd src
uv run python export.py
```

Expected:
```
Modelo exportado a models/plausibility.onnx
Verificación ONNX OK — score de prueba: 0.8731
```

**Step 5: Correr tests**

```bash
uv run pytest tests/test_export.py -v
```

Expected: todos en `PASSED`

**Step 6: Commit final**

```bash
git add src/export.py tests/test_export.py models/plausibility.onnx
git commit -m "feat: export a ONNX con verificación onnxruntime"
```

---

## Task 8: Notebook de Análisis y Visualización

**Files:**
- Create: `notebooks/02_entrenamiento.ipynb`

**Step 1: Iniciar Jupyter**

```bash
uv run jupyter notebook
```

**Step 2: Crear notebook con análisis**

Crear `notebooks/02_entrenamiento.ipynb` con celdas:

```python
# Celda 1: Cargar datos y explorar distribución
import sys
sys.path.insert(0, "../src")
import pandas as pd
import matplotlib.pyplot as plt
from synthea_parser import load_synthea, LOINC_RANGES
from dataset import build_dataset

df_synthea = load_synthea("../data/synthea")
full_df = build_dataset(df_synthea)
print(full_df["label"].value_counts())

# Visualizar distribución de hemoglobina
hb = full_df[full_df["loinc_code"] == "718-7"]
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.hist(hb[hb["label"]==1]["value"], bins=30, color="green", alpha=0.7, label="plausible")
plt.hist(hb[hb["label"]==0]["value"], bins=30, color="red", alpha=0.7, label="implausible")
plt.xlabel("Hemoglobina (g/dL)")
plt.legend()
plt.title("Distribución de valores — Hemoglobina")
plt.show()
```

```python
# Celda 2: Métricas detalladas con sklearn
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import numpy as np
from model import PlausibilityNet
from dataset import PlausibilityDataset, encode_features
from torch.utils.data import DataLoader

# Cargar modelo
model = PlausibilityNet(input_dim=5)
model.load_state_dict(torch.load("../models/plausibility.pt", map_location="cpu"))
model.eval()

# Evaluar en todo el dataset
full_df_enc, _, _ = encode_features(full_df)
ds = PlausibilityDataset(full_df)
loader = DataLoader(ds, batch_size=256)

all_preds, all_labels, all_scores = [], [], []
with torch.no_grad():
    for X, y in loader:
        scores = model(X).squeeze().numpy()
        preds = (scores > 0.5).astype(int)
        all_scores.extend(scores)
        all_preds.extend(preds)
        all_labels.extend(y.numpy().astype(int))

print(classification_report(all_labels, all_preds, target_names=["implausible", "plausible"]))
print(f"AUC-ROC: {roc_auc_score(all_labels, all_scores):.4f}")
```

```python
# Celda 3: Matriz de confusión
import seaborn as sns
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["implausible", "plausible"],
            yticklabels=["implausible", "plausible"])
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.title("Matriz de Confusión")
plt.show()
```

**Step 3: Commit**

```bash
git add notebooks/
git commit -m "feat: notebook de análisis y visualización de métricas"
```

---

## Resultado Final

Al completar todas las tareas tendrás:

```
✅ Entorno Python en M4 con MPS funcionando
✅ Pipeline de datos Synthea + MIMIC-IV
✅ Generación de negativos realistas (errores de conversión + outliers)
✅ MLP entrenado con GPU M4 (MPS)
✅ Test accuracy > 90%
✅ Modelo exportado como plausibility.onnx
✅ Verificado con onnxruntime en Python
✅ Notebook con métricas: F1, AUC-ROC, matriz de confusión
✅ Listo para consumir desde Go con onnxruntime_go
```

**Siguiente proyecto:** Siamese Network para Master Patient Index (MPI)
