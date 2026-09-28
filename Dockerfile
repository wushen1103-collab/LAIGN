FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    aria2 \
    bzip2 \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/LAIGN
COPY environment.yml requirements.txt ./

RUN curl -L --retry 5 \
    https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/Release%2026.3.2-2/Miniforge3-26.3.2-2-Linux-x86_64.sh \
    -o /tmp/miniforge.sh || \
    curl -L --retry 5 \
    https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-x86_64.sh \
    -o /tmp/miniforge.sh

RUN bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda config --system --set channel_priority flexible && \
    conda env create -f environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "laign", "/bin/bash", "-c"]

COPY . .

CMD ["bash"]
