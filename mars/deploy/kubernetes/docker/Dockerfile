ARG BASE_CONTAINER=marsproject/mars-base
FROM ${BASE_CONTAINER}

COPY . /opt/mars/

RUN apt-get -yq update --allow-releaseinfo-change \
  && apt-get -yq install gcc g++ \
  && curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - \
  && sudo apt-get install -y nodejs \
  && /opt/conda/bin/pip install -e /opt/mars \
  && apt-get -yq remove gcc g++ nodejs \
  && apt-get -yq autoremove \
  && apt-get -yq clean \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /usr/local/lib/node_modules
RUN mkdir -p /srv
WORKDIR /srv

RUN cp /opt/mars/mars/deploy/kubernetes/docker/docker-logging.conf /srv/logging.conf \
  && cp /opt/mars/mars/deploy/kubernetes/docker/entrypoint.sh /srv/entrypoint.sh \
  && chmod a+x /srv/*.sh

ENTRYPOINT [ "/srv/entrypoint.sh" ]
