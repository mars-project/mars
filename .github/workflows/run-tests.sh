#!/bin/bash
set -e
if [ -n "$WITH_CYTHON" ]; then
  mkdir -p build
  pytest $PYTEST_CONFIG --cov-config .coveragerc --ignore-glob "*/integrated/*" \
    --ignore mars/tests/test_mutable.py mars/serialize mars/optimizes mars/tests
  mv .coverage build/.coverage.non-fork.file

  export POOL_START_METHOD=forkserver

  retry -n 20 -g INTERNALERROR pytest $PYTEST_CONFIG --cov-config .coveragerc mars/oscar
  mv .coverage build/.coverage.oscar_ctx.file

  pytest $PYTEST_CONFIG --cov-config .coveragerc --forked mars/actors mars/deploy/local \
    mars/scheduler mars/web
  mv .coverage build/.coverage.fork.file
  coverage combine build/ && coverage report
fi
if [ -z "$NO_COMMON_TESTS" ]; then
  mkdir -p build
  pytest $PYTEST_CONFIG --cov-config .coveragerc-threaded mars/tensor mars/dataframe mars/web \
    mars/learn mars/remote mars/storage mars/lib
  mv .coverage build/.coverage.tensor.file
  pytest $PYTEST_CONFIG --cov-config .coveragerc --forked --ignore mars/tensor --ignore mars/dataframe \
    --ignore mars/learn --ignore mars/remote mars
  mv .coverage build/.coverage.main.file
  coverage combine build/ && coverage report

  export DEFAULT_VENV=$VIRTUAL_ENV
  source testenv/bin/activate
  pytest --timeout=1500 mars/tests/test_session.py mars/lib/filesystem/tests/test_filesystem.py
  if [ -z "$DEFAULT_VENV" ]; then
    deactivate
  else
    source $DEFAULT_VENV/bin/activate
  fi
fi
