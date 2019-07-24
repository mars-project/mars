ARG BASE_CONTAINER=ubuntu:bionic-20190612@sha256:9b1702dcfe32c873a770a32cfd306dd7fc1c4fd134adfb783db68defc8894b3c
FROM ${BASE_CONTAINER}

ARG MARS_VERSION

RUN apt-get update -q \
  && apt-get install -yq --no-install-recommends \
    wget \
    build-essential \
    bzip2 \
    ca-certificates \
    git \
    sudo \
    locales \
  && apt-get clean -yq \
  && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
  && wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
  && /bin/bash ~/miniconda.sh -b -p /opt/conda \
  && rm ~/miniconda.sh \
  && /opt/conda/bin/conda install \
    mkl \
    cython \
    gevent \
    numpy \
    scipy \
    numexpr \
    psutil \
    bokeh \
    tornado \
    jinja2 \
    pandas \
    numba \
    lz4 \
  && /opt/conda/bin/conda clean -tipsy
RUN /opt/conda/bin/conda install -c conda-forge \
    libiconv \
    pyarrow \
    tiledb-py \
    python-kubernetes \
  && /opt/conda/bin/conda clean -tipsy

RUN /opt/conda/bin/pip install git+https://github.com/mars-project/mars.git@${MARS_VERSION}
RUN mkdir -p /srv
WORKDIR /srv

ADD docker-logging.conf /srv/logging.conf
ADD entrypoint.sh /srv/entrypoint.sh

ENTRYPOINT [ "/srv/entrypoint.sh" ]
