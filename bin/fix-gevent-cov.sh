#!/bin/bash
GEVENT_DIR=`python -c 'import gevent; print(gevent.__file__)'`
GEVENT_DIR=`dirname $GEVENT_DIR`
if [ ! -z $GEVENT_DIR ]; then
    find $GEVENT_DIR -name *.c -delete
fi
