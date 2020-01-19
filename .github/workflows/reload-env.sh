#!/bin/bash

export UNAME="$(uname | awk '{print tolower($0)}')"

if [[ "$GITHUB_REF" =~ ^"refs/tags/" ]]; then
  export GITHUB_TAG_REF="$GITHUB_REF"
  unset CYTHON_TRACE
  export GIT_TAG=$(echo "$GITHUB_REF" | sed -e "s/refs\/tags\///g")
fi

if [[ $UNAME == "mingw"* ]] || [[ $UNAME == "msys"* ]]; then
  export UNAME="windows"
  CONDA=$(echo "/$CONDA" | sed -e 's/\\/\//g' -e 's/://')
  export PATH="$CONDA/Library:$CONDA/Library/bin:$CONDA/Scripts:$CONDA:$PATH"
  export PATH="$CONDA/envs/test/Library:$CONDA/envs/test/Library/bin:$CONDA/envs/test/Scripts:$CONDA/envs/test:$PATH"
else
  export PATH="$HOME/miniconda/envs/test/bin:$HOME/miniconda/bin:$PATH"
fi
export PYTHON=$(python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:3]))")
