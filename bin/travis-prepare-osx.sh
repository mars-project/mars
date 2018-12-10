#!/bin/bash
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    set -e

    ulimit -n 1024

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    PYENV_ROOT="$HOME/.pyenv"
    PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

    if version_gt "3.0" "$PYTHON" ; then
        curl -O https://bootstrap.pypa.io/get-pip.py
        python get-pip.py --user
    else
        pyenv install $PYTHON
        pyenv global $PYTHON
    fi
fi

#check python version
python -V
