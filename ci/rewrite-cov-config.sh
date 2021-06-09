#!/bin/bash
set -e
if [ -z "$WITH_CYTHON" ]; then
  sed -i.bak "s/plugins *= *Cython\.Coverage//g" setup.cfg;
  sed -i.bak -e '/*\.pxd/ a\
  \ \ \ \ *.pyx \
  ' setup.cfg
else
  export CYTHON_TRACE=1
  sed -i.bak -e '/*\.pxd/ a\
  \ \ \ \ *.py \
  ' setup.cfg
fi
