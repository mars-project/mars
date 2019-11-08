#!/bin/bash
# from https://github.com/jplana/python-etcd/blob/master/download_etcd.sh
# licensed under mit license
set -e
VERSION=${1:-3.3.10}

UNAME="$(uname | awk '{print tolower($0)}')"
if [[ "$UNAME" == "darwin" ]]; then
  realpath() {
      [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
  }
fi

SCRIPT_PATH="$(cd "$(realpath "$(dirname "${BASH_SOURCE[0]}")")" ; pwd -P )"
ETCD_PATH="$(dirname "$(dirname "$SCRIPT_PATH")")/bin"

FILE_EXT=".zip"
if [[ $UNAME == mingw* ]]; then UNAME="windows"; fi
if [[ $UNAME == "linux" ]]; then FILE_EXT=".tar.gz"; fi

mkdir -p $ETCD_PATH
URL="https://github.com/coreos/etcd/releases/download/v${VERSION}/etcd-v${VERSION}-${UNAME}-amd64${FILE_EXT}"
curl -L $URL | tar -C $ETCD_PATH --strip-components=1 -xzvf - "etcd-v${VERSION}-${UNAME}-amd64/etcd"
if [[ $UNAME != "mingw"* ]]; then
  chmod a+x $ETCD_PATH/etcd
fi
export PATH=$PATH:$ETCD_PATH
