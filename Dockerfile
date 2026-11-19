# Dependencies only; source, datasets, caches and results are bind mounted.
ARG BASE_IMAGE=assignment2:torch2.5.1-cu124-v1
FROM ${BASE_IMAGE}
COPY requirements-runtime.txt /opt/genimage-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/genimage-requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /workspace
