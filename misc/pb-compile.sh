#!/bin/bash
(
    cd "$( dirname "${BASH_SOURCE[0]}" )/../"
    for filename in `find mars/serialize -name *.proto`; do
        compiled=${filename%.*}'_pb2.py'
        if [ ! -f $compiled ] || [ $compiled -ot $filename ]; then
            echo "Processing $filename"
            protoc -I. --python_out=. $filename
            if [ ${filename: -13} == "operand.proto" ]; then
                python mars/serialize/protos/genopcodes.py
            fi
        fi
    done
)
