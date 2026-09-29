# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face Spaces — Docker SDK.
# Serves the admin studio (admin_panel:app) behind gunicorn. On a Space uploads,
# training and freezing are disabled, so the hosted copy is results-only.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# libgomp1 is required by scikit-learn / OpenMP-backed wheels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

WORKDIR /home/user/app

# Install CPU-only PyTorch first (no CUDA → much smaller), then the rest.
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt

# Copy the application (model artifacts under outputs/ are included).
COPY --chown=user . .

EXPOSE 7860

# 2 workers is plenty for the expected handful of concurrent users; the long
# timeout covers the first request where large pipelines are loaded from disk.
CMD ["gunicorn", "-w", "2", "--threads", "4", "-t", "180", "-b", "0.0.0.0:7860", "admin_panel:app"]
