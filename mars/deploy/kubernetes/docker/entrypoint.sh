#!/bin/bash
set -e

if [[ "$1" == *"/"* ]]; then
  $@
else
  /opt/conda/bin/python -m "$1" ${@:2}
fi
