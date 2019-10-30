#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# Install requirements
PYBIN=/opt/python/${PYABI}/bin
REQ_FILE=/io/requirements-wheel.txt

# install packages without pandas first
# todo remove this section when dropping support for Python 2.7
REQ_FILE_NO_PD=/tmp/requirements-no-pd.txt
grep -v "pandas" $REQ_FILE > $REQ_FILE_NO_PD
"${PYBIN}/pip" install -r $REQ_FILE_NO_PD

"${PYBIN}/pip" install -r $REQ_FILE

# Compile wheels
cd /io
"${PYBIN}/python" setup.py bdist_wheel


# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

rm dist/*-linux*.whl
