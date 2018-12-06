#!/usr/bin/env bash

make gettext
sphinx-intl update -p build/gettext -l zh_CN
sphinx-build -D language=zh_CN -b html source build/html-zh && make html