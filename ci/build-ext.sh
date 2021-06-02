#!/bin/bash
set -e
if [ -z "$WITH_CYTHON" ]; then
  for cf in `ls .coveragerc*`; do
    sed -i.bak "s/plugins *= *Cython\.Coverage//g" $cf;
    sed -i.bak -e '/*\.pxd/ a\
    \ \ \ \ *.pyx \
    ' $cf
  done
else
  export CYTHON_TRACE=1
  for cf in `ls .coveragerc*`; do
    sed -i.bak -e '/*\.pxd/ a\
    \ \ \ \ *.py \
    ' $cf
  done
fi
retry python setup.py build_ext -i -j 2
pip install -e ".[dev]"
