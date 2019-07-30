#!/bin/bash
RETRIES=$1
shift
for (( RETRY=1; RETRY <= $RETRIES ; RETRY++ )); do
  "$@"
  EXIT=$?
  if [[ $EXIT != 0 ]]; then
    echo "Command attempt $RETRY failed"
  else
    exit 0
  fi
done
exit $EXIT
