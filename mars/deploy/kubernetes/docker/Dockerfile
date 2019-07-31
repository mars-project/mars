ARG BASE_CONTAINER=continuumio/miniconda3:4.6.14
FROM ${BASE_CONTAINER}

ARG MARS_VERSION

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
  && /opt/conda/bin/conda clean -tipsy

RUN apt-get -yq update \
  && apt-get -yq install gcc g++ \
  && /opt/conda/bin/pip install git+https://github.com/mars-project/mars.git@${MARS_VERSION} \
  && apt-get -yq remove gcc g++ \
  && apt-get -yq autoremove \
  && apt-get -yq clean \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /srv
WORKDIR /srv

COPY docker-logging.conf /srv/logging.conf
COPY entrypoint.sh /srv/entrypoint.sh

ENTRYPOINT [ "/srv/entrypoint.sh" ]
