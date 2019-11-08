#!/usr/bin/env python
import glob
import os
import re
import sys


def main(source_dir):
    for fn in glob.glob('.pwd.*'):
        cov_fn = fn.replace('.pwd', '.coverage')
        if not os.path.exists(cov_fn):
            continue

        sys.stderr.write('Rewriting coverage file %s\n' % cov_fn)
        with open(fn, 'r') as f:
            env_pwd = f.read().strip()
        with open(cov_fn, 'r') as f:
            cov_data = f.read().replace(r'\\', '/')

        env_pwd = re.sub('^/([A-Z])/', lambda m: '/' + m.group(1).lower() + '/', env_pwd)
        cov_data = re.sub('"([A-Za-z]):/', lambda m: '"/' + m.group(1).lower() + '/', cov_data)
        cov_data = cov_data.replace(env_pwd, source_dir)

        with open(cov_fn, 'w') as f:
            f.write(cov_data)


if __name__ == '__main__':
    os.chdir(sys.argv[1])
    main(sys.argv[2])
