#!/bin/bash
set -e
if [ -n "$WITH_CYTHON" ]; then
  pytest $PYTEST_CONFIG --cov-config .coveragerc --forked mars/actors mars/deploy/local mars/serialize \
    mars/optimizes mars/scheduler mars/tests mars/web
  coverage report
fi
if [ -z "$NO_COMMON_TESTS" ]; then
  if [[ "$UNAME" == "windows" ]]; then
    export NO_SERIALIZE_IN_TEST_EXECUTOR=1
    python -m pytest $PYTEST_CONFIG --ignore=mars/scheduler --ignore=mars/worker --timeout=1500
    coverage report
  else
    mkdir -p build
    pytest $PYTEST_CONFIG --cov-config .coveragerc-threaded mars/tensor mars/dataframe mars/web mars/learn mars/remote
    mv .coverage build/.coverage.tensor.file
    pytest $PYTEST_CONFIG --cov-config .coveragerc --forked --ignore mars/tensor --ignore mars/dataframe \
      --ignore mars/learn --ignore mars/remote mars
    mv .coverage build/.coverage.main.file
    coverage combine build/ && coverage report

    export DEFAULT_VENV=$VIRTUAL_ENV
    source testenv/bin/activate
    pytest --timeout=1500 mars/tests/test_session.py mars/tests/test_filesystem.py
    if [ -z "$DEFAULT_VENV" ]; then
      deactivate
    else
      source $DEFAULT_VENV/bin/activate
    fi
  fi
fi
