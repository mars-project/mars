#!/bin/bash
set -e -x

if [ "$TRAVIS_TAG" ]; then
  if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    sudo chmod 777 bin/*
    docker pull $DOCKER_IMAGE
    docker run --rm -e "PYVER=$PYVER" -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/bin/travis-build-wheels.sh
  else
    pip wheel --no-deps .
    mkdir dist
    cp *.whl dist/
    pip install delocate
    delocate-wheel dist/*.whl
    delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl
  fi
  ls dist/

  echo "[distutils]"                                  > ~/.pypirc
  echo "index-servers ="                             >> ~/.pypirc
  echo "    pypi"                                    >> ~/.pypirc
  echo "[pypi]"                                      >> ~/.pypirc
  echo "repository=https://upload.pypi.org/legacy/"  >> ~/.pypirc
  echo "username=pyodps"                             >> ~/.pypirc
  echo "password=$PASSWD"                            >> ~/.pypirc

  python -m pip install twine
  python -m twine upload -r pypi --skip-existing dist/*.whl;
else
  echo "Not on a tag, won't deploy to pypi";
fi
