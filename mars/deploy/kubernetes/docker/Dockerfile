ARG BASE_CONTAINER=marsproject/mars-base
FROM ${BASE_CONTAINER}

ARG MARS_VERSION

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
