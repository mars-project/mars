#!/usr/bin/env python
import glob
import os
import re
import sqlite3
import sys


def rewrite_path(txt, old_path=None, new_path=None):
    txt = txt.replace('\\\\', '/').replace('\\', '/')
    txt = re.sub('"([A-Za-z]):/', lambda m: '"/' + m.group(1).lower() + '/', txt)
    txt = re.sub('^([A-Za-z]):/', lambda m: '/' + m.group(1).lower() + '/', txt)
    if old_path is not None and new_path is not None:
        txt = txt.replace(old_path, new_path)
    return txt


def rewrite_json_coverage(file_name, old_path=None, new_path=None):
    with open(file_name, 'r') as f:
        cov_data = rewrite_path(f.read(), old_path, new_path)
    with open(file_name, 'w') as f:
        f.write(cov_data)


def rewrite_sqlite_coverage(file_name, old_path=None, new_path=None):
    conn = None
    try:
        conn = sqlite3.connect(file_name)
        cursor = conn.cursor()
        cursor.execute('SELECT id, path FROM file')

        updates = []
        for file_id, file_path in cursor:
            new_file_path = rewrite_path(file_path, old_path, new_path)
            if file_path != new_file_path:
                updates.append((new_file_path, file_id))

        if updates:
            cursor = conn.cursor()
            cursor.executemany('UPDATE file SET path=? WHERE id=?', updates)
            conn.commit()
    finally:
        if conn is not None:
            conn.close()


def main(source_dir):
    for cov_fn in glob.glob('.coverage*'):
        if 'coveragerc' in cov_fn:
            continue
        sys.stderr.write('Rewriting coverage file %s\n' % cov_fn)

        pwd_fn = cov_fn.replace('.coverage', '.pwd')
        if not os.path.exists(pwd_fn):
            env_pwd = None
        else:
            with open(pwd_fn, 'r') as f:
                env_pwd = f.read().strip()
                env_pwd = re.sub('^/([A-Z])/', lambda m: '/' + m.group(1).lower() + '/', env_pwd)

        with open(cov_fn, 'rb') as f:
            header = f.read(6)

        if header == b'SQLite':
            rewrite_sqlite_coverage(cov_fn, env_pwd, source_dir)
        else:
            rewrite_json_coverage(cov_fn, env_pwd, source_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    source_dir = None
    if len(sys.argv) > 2:
        source_dir = sys.argv[2]
    main(source_dir)
