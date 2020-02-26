#!/bin/bash
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

PYTHON=$(cut -d '-' -f 1 <<< "$PYTHON")
UNAME="$(uname | awk '{print tolower($0)}')"
FILE_EXT="sh"
if [[ "$UNAME" == "darwin" ]]; then
  set -e
  ulimit -n 1024
  CONDA_OS="MacOSX"
elif [[ $UNAME == "linux" ]]; then
  sudo apt-get install -y liblz4-dev
  CONDA_OS="Linux"
elif [[ $UNAME == "mingw"* ]] || [[ $UNAME == "msys"* ]]; then
  CONDA_OS="Windows"
  FILE_EXT="exe"
  # VS2015 needed by Python 3.5
  if [[ "$PYTHON" == "3.5" ]] || [[ "$PYTHON" == "3.6" ]]; then
    vcbindir="/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/BIN/x86_amd64"
    if [[ ! -d $vcbindir ]]; then
      npm install --global --silent --vs2015 windows-build-tools
    fi
    # Remove SDK of Windows Driver Framework to avoid confusion
    rm -rf "/c/Program Files (x86)/Windows Kits/10/include/wdf"
    # Locate and copy resource compiler into VC dir
    for subdir in $(ls -r "/c/Program Files (x86)/Windows Kits/10/bin"); do
      bindir="/c/Program Files (x86)/Windows Kits/10/bin/$subdir/x64"
      if [ -f "$bindir/rc.exe" ] && [ -f "$bindir/rcdll.dll" ]; then
        echo "Copying rc.exe from $bindir"
        cp "$bindir/rc.exe" "$vcbindir"
        cp "$bindir/rcdll.dll" "$vcbindir"
        break
      fi
    done
  fi
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
