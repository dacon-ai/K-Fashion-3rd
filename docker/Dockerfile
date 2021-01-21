
ARG CUDA_VERSION_MAJOR_MINOR

FROM nvidia/cuda:${CUDA_VERSION_MAJOR_MINOR}-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
    apt-get update -qq && apt-get install -yq --no-install-recommends \
        build-essential git curl wget cmake vim ssh bzip2 ca-certificates \
        ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# https://hub.docker.com/r/continuumio/miniconda3/dockerfile
RUN \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# https://plotly.com/python/getting-started/
#RUN \
#    conda install -c plotly plotly=4.8.2 && \
#    conda install nodejs jupyterlab "ipywidgets=7.5" && \
#    jupyter labextension install jupyterlab-plotly@4.8.2 && \
#    jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.8.2

RUN conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
RUN conda install pandas

RUN pip install mmcv-full==latest+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

#RUN conda clean --all
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
#WORKDIR /mmdetection
#RUN pip install -r requirements/build.txt
#RUN pip install --no-cache-dir -e .

# environment variables
ENV LC_ALL=C.UTF-8
ENV TZ="Asia/Seoul"
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64/
