#!/bin/bash
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

PYTHON=$(cut -d '-' -f 1 <<< "$PYTHON")
UNAME="$(uname | awk '{print tolower($0)}')"
FILE_EXT="sh"
if [[ "$UNAME" == "darwin" ]]; then
    ulimit -n 1024
    CONDA_OS="MacOSX"
elif [[ $UNAME == "linux" ]]; then
    sudo apt-get install -y liblz4-dev
    CONDA_OS="Linux"
elif [[ $UNAME == "mingw"* ]]; then
    CONDA_OS="Windows"
    FILE_EXT="exe"
fi

if version_gt "3.0" "$PYTHON" ; then
    CONDA_FILE="Miniconda2-latest-${CONDA_OS}-x86_64.${FILE_EXT}"
else
    CONDA_FILE="Miniconda3-latest-${CONDA_OS}-x86_64.${FILE_EXT}"
fi

TEST_PACKAGES="virtualenv gevent psutil pyyaml lz4"

if [[ "$FILE_EXT" == "sh" ]]; then
  curl -s -o "miniconda.${FILE_EXT}" https://repo.continuum.io/miniconda/$CONDA_FILE
  bash miniconda.sh -b -p $HOME/miniconda && rm -f miniconda.*
  CONDA_BIN_PATH=$HOME/miniconda/bin
  TEST_PACKAGES="$TEST_PACKAGES nomkl libopenblas"
  export PATH="$HOME/miniconda/envs/test/bin:$HOME/miniconda/bin:$PATH"
else
  CONDA=$(echo "/$CONDA" | sed -e 's/\\/\//g' -e 's/://')
  echo "Using installed conda at $CONDA"
  CONDA_BIN_PATH=$CONDA/Scripts
  export PATH="$CONDA/envs/test/Scripts:$CONDA/envs/test:$CONDA/Scripts:$CONDA:$PATH"
fi
$CONDA_BIN_PATH/conda create --quiet --yes -n test python=$PYTHON $TEST_PACKAGES

#check python version
export PYTHON=$(python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:3]))")
echo "Installed Python version: $PYTHON"
