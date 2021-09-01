#!/bin/bash
for fn in $(git diff --name-only master); do
  autopep8 --in-place $fn
done
