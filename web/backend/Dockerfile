# KhanhLLM Backend — Multi-stage Docker build
# See docs/06-web-app-design.md for deployment design.
#
# Build:
#   docker build -t khanh-llm-backend ./web/backend
#
# Run (mount model checkpoint from host):
#   docker run --gpus all \
#     -v /path/to/runs:/runs \
#     -p 8000:8000 \
#     khanh-llm-backend

FROM nvcr.io/nvidia/pytorch:24.03-py3 AS base

WORKDIR /app

# Install the khanh_llm package (from repo root)
COPY ../../pyproject.toml ../../README.md /khanh_llm_src/
COPY ../../src /khanh_llm_src/src
RUN pip install -e /khanh_llm_src

# Install backend dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy app source
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
