#!/bin/bash
set -e
PYTEST_CONFIG="--log-level=DEBUG --cov-report= --cov=mars --timeout=1500 -W ignore::PendingDeprecationWarning
--ignore mars/lib/functools32 --ignore mars/lib/futures"
if [ -n "$WITH_HDFS" ]; then
  pytest $PYTEST_CONFIG --cov-config .coveragerc-threaded mars/dataframe/datasource/tests/test_hdfs.py
  coverage report
fi
if [ -n "$WITH_KUBERNETES" ]; then
  pytest $PYTEST_CONFIG --cov-config .coveragerc --forked mars/deploy/kubernetes
  coverage report
fi
if [ -z "$NO_COMMON_TESTS" ]; then
  if [[ "$UNAME" == "mingw"* ]]; then
    python -m pytest $PYTEST_CONFIG --ignore=mars/scheduler --ignore=mars/worker --timeout=1500
    coverage report
  else
    mkdir -p build
    pytest $PYTEST_CONFIG --cov-config .coveragerc-threaded mars/tensor mars/dataframe mars/web mars/learn
    mv .coverage build/.coverage.tensor.file
    pytest $PYTEST_CONFIG --cov-config .coveragerc --forked --ignore mars/tensor --ignore mars/dataframe \
     --ignore mars/learn mars
    mv .coverage build/.coverage.main.file
    coverage combine build/ && coverage report --fail-under=85

    export DEFAULT_VENV=$VIRTUAL_ENV
    source testenv/bin/activate
    pytest --timeout=1500 mars/tests/test_session.py
    if [ -z "$DEFAULT_VENV" ]; then
      deactivate
    else
      source $DEFAULT_VENV/bin/activate
    fi
  fi
fi
