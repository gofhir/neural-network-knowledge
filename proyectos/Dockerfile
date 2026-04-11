FROM python:3.11-slim

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir tensorflow && \
    pip install --no-cache-dir jax flax && \
    pip install --no-cache-dir matplotlib

RUN mkdir -p /app/output

WORKDIR /app

COPY clase7_ejemplos/ ./ejemplos/
COPY clase8_ejemplos/ ./ejemplos_c8/

CMD ["python", "-u", "ejemplos/pytorch/01_normalizacion_manual_pytorch.py"]
