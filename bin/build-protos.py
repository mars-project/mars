#!/usr/bin/env python
import os
import shutil
import subprocess
import stat
import sys
import tempfile
import zipfile
from distutils.spawn import find_executable

protobuf_version = '3.6.0'
_protoc_executable = None
_temp_protoc_path = None


def execfile(fname, globs, locs=None):
    locs = locs or globs
    exec(compile(open(fname).read(), fname, "exec"), globs, locs)


def _get_protoc_executable():
    global _protoc_executable

    if _protoc_executable:
        return _protoc_executable

    _protoc_executable = find_executable('protoc') or os.environ.get('PROTOC')
    if _protoc_executable is None:
        sys.stderr.write('Cannot find protoc. Trying to download.\n')
        _protoc_executable = _download_protoc_executable()
    return _protoc_executable


def _download_protoc_executable():
    global _temp_protoc_path
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    protoc_bin = 'protoc'
    if sys.platform == 'win32':
        protoc_package = 'protoc-%s-win32.zip' % protobuf_version
        protoc_bin = 'protoc.exe'
    elif sys.platform == 'darwin':
        protoc_package = 'protoc-%s-osx-x86_64.zip' % protobuf_version
    else:
        protoc_package = 'protoc-%s-linux-x86_64.zip' % protobuf_version
    protoc_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v%s/%s' \
                 % (protobuf_version, protoc_package)
    sys.stderr.write('Downloading protoc from %s. You can download it manually, '
                     'extract the downloaded package and then specify the '
                     'environment variable PROTOC as the path to the protoc '
                     'binary and run the command again.' % protoc_url)

    temp_path = _temp_protoc_path = tempfile.mkdtemp(prefix='mars-setup-')
    zip_path = os.path.join(temp_path, 'protoc.zip')
    urlretrieve(protoc_url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as protoc_zip:
        executable = protoc_zip.extract('bin/%s' % protoc_bin, temp_path)
        os.chmod(executable, os.stat(executable).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return executable


def main(repo_root):
    for root, _, files in os.walk(repo_root):
        for fn in files:
            if not fn.endswith('.proto'):
                continue

            src_fn = os.path.join(root, fn)
            compiled_fn = src_fn.replace('.proto', '_pb2.py')
            rel_path = os.path.relpath(src_fn, repo_root)
            if not os.path.exists(compiled_fn) or \
                    os.path.getmtime(src_fn) > os.path.getmtime(compiled_fn):
                sys.stdout.write('compiling protobuf %s\n' % rel_path)
                subprocess.check_call([_get_protoc_executable(), '--python_out=.', rel_path],
                                      cwd=repo_root)

            if fn == 'operand_type.proto':
                op_globs = dict()
                opcode_fn = os.path.join(repo_root, 'mars', 'opcodes.py')
                sys.stdout.write('constructing opcodes with %s\n' % rel_path)
                if not os.path.exists(opcode_fn) or \
                        os.path.getmtime(src_fn) > os.path.getmtime(opcode_fn):
                    try:
                        import google.protobuf
                    except ImportError:
                        sys.stderr.write('You need to install protobuf to build proto files.\n')
                        sys.exit(1)

                    execfile(compiled_fn, op_globs)
                    OperandType = op_globs['OperandType']
                    with open(os.path.join(repo_root, 'mars', 'opcodes.py'), 'w') as opcode_file:
                        opcode_file.write('# Generated automatically from %s.  DO NOT EDIT!\n' % rel_path)
                        for val, desc in OperandType.DESCRIPTOR.values_by_number.items():
                            opcode_file.write('{0} = {1!r}\n'.format(desc.name, val))
    if _temp_protoc_path:
        shutil.rmtree(_temp_protoc_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        repo_root = sys.argv[1]
    else:
        repo_root = os.path.curdir
    main(repo_root)
