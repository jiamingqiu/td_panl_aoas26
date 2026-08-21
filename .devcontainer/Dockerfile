# Convenience environment only; this is not the original cluster environment.
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-venv \
        r-base-core \
        r-cran-lubridate \
        r-cran-tidyverse \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements-python.txt /tmp/requirements-python.txt

RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.0 \
        setuptools==69.5.1 \
        wheel==0.43.0 \
    && python -m pip install --no-cache-dir \
        torch==2.3.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir \
        -r /tmp/requirements-python.txt \
    && rm -f /tmp/requirements-python.txt \
    && chown -R vscode:vscode /opt/venv
