#!/bin/bash
set -e
cd /mnt/mars
mkdir -p .kube-coverage
export COVERAGE_FILE=.kube-coverage/.coverage

cp /mnt/mars/.coveragerc /tmp/coveragerc
export COVERAGE_PROCESS_START=/tmp/coveragerc

COV_RUNNER="/opt/conda/bin/coverage run --rcfile=/tmp/coveragerc"

if [[ $1 == *"probe"* ]]; then
  PROBED_FILE=/tmp/.probe.covered.$MARS_K8S_POD_NAME
  if [[ -f "$PROBED_FILE" ]]; then
    /opt/conda/bin/python -m "$1" ${@:2}
  else
    $COV_RUNNER -m "$1" ${@:2}
    touch "$PROBED_FILE"
  fi
else
  if [[ $1 == *"scheduler"* ]]; then
    $COV_RUNNER -m "$1" -Dscheduler.default_cpu_usage=0 --log-conf /srv/logging.conf ${@:2}
  elif [[ $1 == *"worker"* ]]; then
    $COV_RUNNER -m "$1" --ignore-avail-mem --cache-mem 64m --log-conf /srv/logging.conf ${@:2}
  else
    $COV_RUNNER -m "$1" --log-conf /srv/logging.conf ${@:2}
  fi
  while [[ -f /tmp/stopping.tmp ]]; do
    sleep 1
  done
fi
