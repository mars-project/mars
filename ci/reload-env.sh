#!/bin/bash

export UNAME="$(uname | awk '{print tolower($0)}')"
export PYTEST_CONFIG="--log-level=DEBUG --cov-report= --cov=mars --timeout=1500 -W ignore::PendingDeprecationWarning"

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
  retrial=5
  unset grep_err_str
  # parse parameters
  while true; do
    if [ $1 == "-n" ]; then
      retrial=$2
      shift; shift
    elif [ $1 == "-g" ]; then
      grep_err_str=$2
      shift; shift
    else
      break
    fi
  done

  # do retry
  r=0
  std_tmp_dir=$(mktemp -d -t retry_log.XXXXX)
  while true; do
    r=$((r+1))
    stdout_file_name="$std_tmp_dir/stdout.$r.log"
    stderr_file_name="$std_tmp_dir/stderr.$r.log"

    unset ret
    if [ -z $grep_err_str ]; then
      "$@" || ret=$?
    else
      touch "$stdout_file_name"
      touch "$stderr_file_name"
      "$@" > >(tee "$stdout_file_name") 2> >(tee "$stderr_file_name" >&2) || ret=$?
    fi
    if [ -z $ret ]; then ret=0; fi

    if [ "$r" -ge $retrial ]; then
      rm -rf "$std_tmp_dir" || true
      return $ret
    else
      if [ $ret -eq 0 ]; then
        rm -rf "$std_tmp_dir" || true
        return 0
      elif [ -n $grep_err_str ]; then
        if grep -q $grep_err_str "$stdout_file_name" || grep -q $grep_err_str "$stderr_file_name"; then
          :
        else
          rm -rf "$std_tmp_dir" || true
          return $ret
        fi
      fi
      sleep 1
    fi
  done
}
alias pip="retry pip"
shopt -s expand_aliases
