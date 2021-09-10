ARG BASE_CONTAINER=continuumio/miniconda3:4.9.2
FROM ${BASE_CONTAINER}

COPY retry.sh /srv/retry.sh

RUN /srv/retry.sh 3 /opt/conda/bin/conda install \
    cloudpickle \
    cython \
    greenlet \
    mkl \
    numba \
    numexpr \
    numpy\>=1.14.0 \
    pandas\>=1.0.0 \
    psutil \
    scikit-learn \
    scipy \
    sqlalchemy \
    tornado \
    lz4 \
  && /srv/retry.sh 3 /opt/conda/bin/conda install -c conda-forge \
    libiconv \
    pyarrow\>=1.0 \
    tiledb-py \
    python-kubernetes \
    uvloop \
  && /opt/conda/bin/conda clean --all -f -y

RUN apt-get -yq update --allow-releaseinfo-change \
  && apt-get -yq install curl sudo procps \
  && apt-get -yq clean \
  && rm -rf /var/lib/apt/lists/* \
