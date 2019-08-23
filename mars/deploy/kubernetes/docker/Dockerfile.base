ARG BASE_CONTAINER=continuumio/miniconda3:4.6.14
FROM ${BASE_CONTAINER}

COPY retry.sh /srv/retry.sh

RUN /srv/retry.sh 3 /opt/conda/bin/conda install \
    bokeh \
    cython \
    gevent \
    jinja2 \
    lz4 \
    mkl \
    numba \
    numexpr \
    numpy \
    pandas \
    protobuf \
    psutil \
    scipy \
    tornado \
  && /srv/retry.sh 3 /opt/conda/bin/conda install -c conda-forge \
    libiconv \
    pyarrow \
    tiledb-py \
    python-kubernetes \
  && /opt/conda/bin/conda clean --all -f -y
