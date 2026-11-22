# HistoGate — Hugging Face Space (FastAPI + MCP on one port, 7860).
# CPU-only image: the default index resolves torch to CUDA wheels (~2.5 GB of
# nvidia-* libs a free CPU Space never touches), so torch/torchvision/nvidia/
# triton are exported OUT of the lock and reinstalled from the CPU index.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libopenslide0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv venv .venv \
 && uv export --frozen --no-dev --no-emit-package torch \
      --no-emit-package torchvision \
      | grep -vE '^(nvidia-|triton)' > /tmp/req.txt \
 && uv pip install --python .venv/bin/python -r /tmp/req.txt \
 && uv pip install --python .venv/bin/python torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY MODEL_CARD.md ./
COPY docs ./docs
COPY data/demo ./data/demo
COPY runs ./runs
COPY mlflow_public ./mlflow_public
RUN uv pip install --python .venv/bin/python --no-deps .

ENV HISTOGATE_SLIDE=data/demo/demo_slide.tif
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 7860
CMD ["histogate-serve", "--space", "0.0.0.0:7860"]
