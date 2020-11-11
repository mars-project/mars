#!/bin/bash
set -e

if [[ $MARS_K8S_REMOUNT_SHM ]]; then
    if [[ -z "$MARS_CACHE_MEM_SIZE" ]]; then
        phymem=$(free|awk '/^Mem:/{print $2}')
        MARS_CACHE_MEM_SIZE=$((phymem * 1000 / 2 ))
    fi
    sudo mount -o remount,size=$MARS_CACHE_MEM_SIZE /dev/shm
fi

if [[ "$1" == *"/"* ]]; then
  $@
else
  /opt/conda/bin/python -m "$1" ${@:2}
fi
