#!/bin/bash
if git diff origin/master -- mars | grep -q "old mode"; then
  echo "Unexpected file mode changed. You may call"
  echo "    git config core.fileMode false"
  echo "before committing to the repo."
  git diff origin/master -- mars | grep -B 2 -A 2 "old mode"
  exit 1
fi
