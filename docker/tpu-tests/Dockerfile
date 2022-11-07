FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/master

ARG PYTHON_VERSION=3.7

ARG PYTORCH_VERSION=1.8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y --name container python=$PYTHON_VERSION

# Run the rest of commands within the new conda env.
# Use absolute path to appease Codefactor.
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "container", "/bin/bash", "-c"]
RUN conda install -y python=$PYTHON_VERSION mkl

RUN pip uninstall -y torch && \
    # Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m
    gsutil cp 'gs://tpu-pytorch/wheels/torch-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' . && \
    gsutil cp 'gs://tpu-pytorch/wheels/torch_xla-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' . && \
    pip install 'torch-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    pip install 'torch_xla-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    rm 'torch-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    rm 'torch_xla-${PYTORCH_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    apt-get install -y libomp5

ENV LD_LIBRARY_PATH=root/miniconda3/envs/container/lib

# Install kornia/kornia at the current PR, plus dependencies.
RUN git clone https://github.com/kornia/kornia.git && \
    cd kornia && \
    git fetch origin $GITHUB_REF:CI && \
    git checkout CI && \
    python setup.py install && \
    pip install pytest

RUN conda init bash

RUN python -c "import torch; print(torch.__version__)"
RUN python -c "import torch_xla; print(torch_xla.__version__)"
RUN python -c "import kornia as K; print(K.__version__)"
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
