#!/bin/bash
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    set -e

    ulimit -n 1024

    function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

    if version_gt "3.0" "$PYTHON" ; then
        curl -O https://bootstrap.pypa.io/get-pip.py
        python get-pip.py --user
    else
        curl -s -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
        $HOME/miniconda/bin/conda create --quiet --yes -n test python=$PYTHON virtualenv gevent psutil pyyaml nomkl
        export PATH="$HOME/miniconda/envs/test/bin:$PATH"
    fi
fi

#check python version
python -V
