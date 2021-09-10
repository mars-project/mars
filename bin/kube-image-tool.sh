#!/bin/bash
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)"
ROOT_PATH=$(dirname "$SCRIPT_PATH")

function usage() {
  echo "Usage: $0 [-r <repo>] [-t <tag>] [-o <organization>] (build | push)"
}

function image_ref() {
  if [[ -z "$TAG" ]]; then
    TAG="v$(python -c "import mars; print(mars.__version__)")"
  fi
  if [[ -z "$ORG" ]]; then
    ORG="marsproject"
  fi
  local image_name="$ORG/mars:$TAG"
  if [[ -n "$REPO" ]]; then
    local image_name="$REPO/$image_name"
  fi
  echo $image_name
}

function build() {
  (
    cd "$ROOT_PATH" || exit 1
    docker build -f mars/deploy/kubernetes/docker/Dockerfile -t "$(image_ref)" .
  )
}

function push() {
  docker push "$(image_ref)"
}

if [[ "$@" = *--help ]] || [[ "$@" = *-h ]] || [[ "$@" = *-\? ]]; then
  usage
  exit 0
fi

while getopts ":r:t:o:" option; do
  case "$option" in
    r)
      REPO=$OPTARG
      ;;
    t)
      TAG=$OPTARG
      ;;
    o)
      ORG=$OPTARG
      ;;
    *)
      usage
      exit 1
  esac
done

case "${@: -1}" in
  build)
    build
    ;;
  push)
    if [[ -z "$REPO" ]]; then
      usage
      exit 1
    fi
    push
    ;;
  *)
    usage
    exit 1
esac
