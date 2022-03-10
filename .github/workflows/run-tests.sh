#!/bin/bash
set -e
if [ -n "$WITH_CYTHON" ]; then
  mkdir -p build
  export POOL_START_METHOD=forkserver

  coverage run --rcfile=setup.cfg -m pytest $PYTEST_CONFIG_WITHOUT_COV mars/tests mars/core/graph
  python .github/workflows/remove_tracer_errors.py
  coverage combine
  mv .coverage build/.coverage.non-oscar.file

  coverage run --rcfile=setup.cfg -m pytest $PYTEST_CONFIG_WITHOUT_COV mars/oscar
  python .github/workflows/remove_tracer_errors.py
  coverage combine
  mv .coverage build/.coverage.oscar_ctx.file

  coverage combine build/ && coverage report
fi
if [ -z "$NO_COMMON_TESTS" ]; then
  mkdir -p build
  pytest $PYTEST_CONFIG mars/remote mars/storage mars/lib mars/metrics
  mv .coverage build/.coverage.tileable.file

  pytest $PYTEST_CONFIG --forked --ignore mars/tensor --ignore mars/dataframe \
    --ignore mars/learn --ignore mars/remote mars
  mv .coverage build/.coverage.main.file
  coverage combine build/ && coverage report
fi
