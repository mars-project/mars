#!/bin/bash
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    set -e
    ulimit -n 1024
    CONDA_OS="MacOSX"
elif [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    CONDA_OS="Linux"
fi

if version_gt "3.0" "$PYTHON" ; then
    CONDA_FILE="Miniconda2-latest-${CONDA_OS}-x86_64.sh"
else
    CONDA_FILE="Miniconda3-latest-${CONDA_OS}-x86_64.sh"
fi

curl -s -o miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE
bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
$HOME/miniconda/bin/conda create --quiet --yes -n test python=$PYTHON virtualenv gevent psutil pyyaml nomkl
export PATH="$HOME/miniconda/envs/test/bin:$HOME/miniconda/bin:$PATH"

#check python version
python -V
