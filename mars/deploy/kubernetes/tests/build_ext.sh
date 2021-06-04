#!/bin/bash
set -e
cd /mnt/mars
/opt/conda/bin/python setup.py build_ext -i
