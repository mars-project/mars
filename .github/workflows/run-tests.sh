#!/bin/bash
set -e
if [ -n "$WITH_CYTHON" ]; then
  mkdir -p build
  export POOL_START_METHOD=forkserver

  echo "aaaaaa"
  coverage run --rcfile=setup.cfg -m pytest $PYTEST_CONFIG_WITHOUT_COV -s -v --log-cli-level=debug \
    mars/tests \
    mars/core/graph \
    mars/serialization
  echo "bbbbbb"
  python .github/workflows/remove_tracer_errors.py
  echo "cccccc"
  coverage combine
  echo "dddddd"
  mv .coverage build/.coverage.non-oscar.file
  echo "eeeeee"

  coverage run --rcfile=setup.cfg -m pytest $PYTEST_CONFIG_WITHOUT_COV -s -v --log-cli-level=debug mars/oscar
  echo "ffffff"
  python .github/workflows/remove_tracer_errors.py
  echo "gggggg"
  coverage combine
  echo "hhhhhh"
  mv .coverage build/.coverage.oscar_ctx.file
  echo "iiiiii"

  coverage combine build/ && coverage report
  echo "jjjjjj"
fi
if [ -z "$NO_COMMON_TESTS" ]; then
  mkdir -p build
  echo "111111"
  pytest $PYTEST_CONFIG -s -v --log-cli-level=debug mars/remote mars/storage mars/lib mars/metrics
  echo "222222"
  mv .coverage build/.coverage.tileable.file
  echo "333333"

  pytest $PYTEST_CONFIG -s -v --log-cli-level=debug --forked --ignore mars/tensor --ignore mars/dataframe \
    --ignore mars/learn --ignore mars/remote mars
  echo "444444"
  mv .coverage build/.coverage.main.file
  echo "555555"
  coverage combine build/ && coverage report
  echo "666666"
fi
