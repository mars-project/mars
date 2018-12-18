#!/bin/bash

if [ "$TRAVIS_TAG" ]; then
  echo "[distutils]"                                  > ~/.pypirc
  echo "index-servers ="                             >> ~/.pypirc
  echo "    pypi"                                    >> ~/.pypirc
  echo "[pypi]"                                      >> ~/.pypirc
  echo "repository=https://upload.pypi.org/legacy/"  >> ~/.pypirc
  echo "username=pyodps"                             >> ~/.pypirc
  echo "password=$PASSWD"                            >> ~/.pypirc

  python -m pip install twine auditwheel

  python setup.py bdist_wheel

  for whl in dist/*.whl; do
	python -m auditwheel repair $whl -w dist/
  done
  rm dist/*-linux*.whl

  python -m twine upload -r pypi --skip-existing dist/*.whl;
else
  echo "Not on a tag, won't deploy to pypi";
fi
