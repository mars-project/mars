#!/bin/bash
set -e

if [ -z "$GITHUB_TAG_REF" ]; then
  echo "Not on a tag, won't deploy to pypi"
elif [ -n "$NO_DEPLOY" ]; then
  echo "Not on a build config, won't deploy to pypi"
else
  git clean -f -x
  source activate test

  if [ -n "$BUILD_STATIC" ]; then
    python setup.py sdist --formats=gztar
  fi

  if [ "$UNAME" = "linux" ]; then
    sudo chmod 777 bin/*
    docker pull $DOCKER_IMAGE

    pyabis=$(echo $PYABI | tr ":" "\n")
    for abi in $pyabis; do
      git clean -f -x
      docker run --rm -e "PYABI=$abi" -e "GIT_TAG=$GIT_TAG" -v `pwd`:/io \
        $DOCKER_IMAGE $PRE_CMD /io/.github/workflows/build-wheels.sh
    done

  else
    conda create --quiet --yes -n wheel python=$PYTHON
    conda activate wheel

    pip install -r requirements-wheel.txt
    pip wheel --no-deps .

    conda activate test

    mkdir -p dist
    cp *.whl dist/

    if [[ "$UNAME" == "darwin" ]]; then
      pip install delocate
      delocate-wheel dist/*.whl
      delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl
    fi
  fi
  ls dist/

  if [[ "$GITHUB_REPOSITORY" == "mars-project/mars" ]]; then
    PYPI_REPO="https://upload.pypi.org/legacy/"
  else
    PYPI_REPO="https://test.pypi.org/legacy/"
  fi

  echo "[distutils]"             > ~/.pypirc
  echo "index-servers ="        >> ~/.pypirc
  echo "    pypi"               >> ~/.pypirc
  echo "[pypi]"                 >> ~/.pypirc
  echo "repository=$PYPI_REPO"  >> ~/.pypirc
  echo "username=pyodps"        >> ~/.pypirc
  echo "password=$PYPI_PWD"     >> ~/.pypirc

  python -m pip install twine
  python -m twine upload -r pypi --skip-existing dist/*.whl
fi
