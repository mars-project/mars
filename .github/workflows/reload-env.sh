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
  export CONDA="$HOME/miniconda"
  export PATH="$HOME/miniconda/envs/test/bin:$HOME/miniconda/bin:$PATH"
fi

if [ -n "$WITH_HADOOP" ] && [ -d /usr/local/hadoop ]; then
  export HADOOP_HOME="/usr/local/hadoop"
  export HADOOP_INSTALL=$HADOOP_HOME
  export HADOOP_MAPRED_HOME=$HADOOP_HOME
  export HADOOP_COMMON_HOME=$HADOOP_HOME
  export HADOOP_HDFS_HOME=$HADOOP_HOME
  export YARN_HOME=$HADOOP_HOME
  export HADOOP_COMMON_LIB_NATIVE_DIR="$HADOOP_HOME/lib/native"
  export PATH="$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin"
fi

export PYTHON=$(python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:3]))")

function retry {
  r=0
  until [ "$r" -ge 5 ]; do
    $@ && break
    r=$((r+1))
    sleep 1
  done
}
