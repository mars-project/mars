#!/bin/bash
# from https://github.com/jplana/python-etcd/blob/master/download_etcd.sh
# licensed under mit license
set -e
VERSION=${1:-3.3.10}
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
UNAME=`uname | awk '{print tolower($0)}'`
FILE_EXT=".zip"
if [[ $UNAME == mingw* ]]; then UNAME="windows"; fi
if [[ $UNAME == "linux" ]]; then FILE_EXT=".tar.gz"; fi

mkdir -p bin
URL="https://github.com/coreos/etcd/releases/download/v${VERSION}/etcd-v${VERSION}-${UNAME}-amd64${FILE_EXT}"
curl -L $URL | tar -C $SCRIPT_PATH --strip-components=1 -xzvf - "etcd-v${VERSION}-${UNAME}-amd64/etcd"
echo $SCRIPT_PATH
export PATH=$PATH:$SCRIPT_PATH
