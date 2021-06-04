#!/bin/bash
set -e
cd /mnt/mars
/opt/conda/bin/pip install -r requirements.txt
/opt/conda/bin/pip install -r requirements-extra.txt

mkdir -p .dist-coverage
export COVERAGE_FILE=.dist-coverage/.coverage

cp /mnt/mars/.coveragerc /tmp/coveragerc
export COVERAGE_PROCESS_START=/tmp/coveragerc

COV_RUNNER="/opt/conda/bin/coverage run --rcfile=/tmp/coveragerc"

if [[ $1 == *"supervisor"* ]]; then
  $COV_RUNNER -m "$1" -f /srv/config.yml --log-conf /srv/logging.conf ${@:2}
elif [[ $1 == *"worker"* ]]; then
  $COV_RUNNER -m "$1" -f /srv/config.yml --log-conf /srv/logging.conf ${@:2}
else
  $COV_RUNNER -m "$1" --log-conf /srv/logging.conf ${@:2}
fi
while [[ -f /tmp/stopping.tmp ]]; do
  sleep 1
done
