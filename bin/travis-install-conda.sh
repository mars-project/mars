#!/bin/bash
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    set -e
    ulimit -n 1024
    CONDA_OS="MacOSX"
elif [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    sudo apt-get install -y liblz4-dev
    CONDA_OS="Linux"
fi

if version_gt "3.0" "$PYTHON" ; then
    CONDA_FILE="Miniconda2-latest-${CONDA_OS}-x86_64.sh"
else
    CONDA_FILE="Miniconda3-latest-${CONDA_OS}-x86_64.sh"
fi

curl -s -o miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE
bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
if [[ "$PYTHON" == "3.7" ]]; then
  PYTHON = "3.7.3"
fi
$HOME/miniconda/bin/conda create --quiet --yes -n test python=$PYTHON virtualenv gevent psutil \
    pyyaml nomkl libopenblas lz4
export PATH="$HOME/miniconda/envs/test/bin:$HOME/miniconda/bin:$PATH"

#check python version
export PYTHON=$(python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:3]))")
echo "Installed Python version: $PYTHON"
