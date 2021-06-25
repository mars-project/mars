#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import functools
import operator
import os
import warnings
import threading
from copy import deepcopy
from typing import Union, Dict


_DEFAULT_REDIRECT_WARN = 'Option {source} has been replaced by {target} and ' \
                         'might be removed in a future release.'


class OptionError(Exception):
    pass


class Redirection:
    def __init__(self, item, warn=None):
        self._items = item.split('.')
        self._warn = warn
        self._warned = True
        self._parent = None

    def bind(self, attr_dict):
        self._parent = attr_dict
        self.getvalue()
        self._warned = False

    def getvalue(self):
        if self._warn and not self._warned:
            self._warned = True
            warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items:
            conf = getattr(conf, it)
        return conf

    def setvalue(self, value):
        if self._warn and not self._warned:
            self._warned = True
            warnings.warn(self._warn)
        conf = self._parent.root
        for it in self._items[:-1]:
            conf = getattr(conf, it)
        setattr(conf, self._items[-1], value)


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        self._inited = False
        self._parent = kwargs.pop('_parent', None)
        self._root = None
        super().__init__(*args, **kwargs)
        self._inited = True

    @property
    def root(self):
        if self._root is not None:
            return self._root
        if self._parent is None:
            self._root = self
        else:
            self._root = self._parent.root
        return self._root

    def __getattr__(self, item):
        if item in self:
            val = self[item]
            if isinstance(val, AttributeDict):
                return val
            elif isinstance(val[0], Redirection):
                return val[0].getvalue()
            else:
                return val[0]
        return object.__getattribute__(self, item)

    def __dir__(self):
        return list(self.keys())

    def register(self, key, value, validator=None):
        if isinstance(validator, tuple):
            validator = any_validator(*validator)
        self[key] = value, validator
        if isinstance(value, Redirection):
            value.bind(self)

    def unregister(self, key):
        del self[key]

    def _setattr(self, key, value, silent=False):
        splits = key.split('.')
        target = self
        for k in splits[:-1]:
            if not silent and (not isinstance(target, AttributeDict) or k not in target):
                raise OptionError('You can only set the value of existing options')
            target = target[k]
        key = splits[-1]

        if not isinstance(value, AttributeDict):
            validate = None
            if key in target:
                val = target[key]
                validate = target[key][1]
                if validate is not None:
                    if not validate(value):
                        raise ValueError(f'Cannot set value `{value}`')
                if isinstance(val[0], Redirection):
                    val[0].setvalue(value)
                else:
                    target[key] = value, validate
            else:
                target[key] = value, validate
        else:
            target[key] = value

    def __setattr__(self, key, value):
        if key == '_inited':
            super().__setattr__(key, value)
            return
        try:
            object.__getattribute__(self, key)
            super().__setattr__(key, value)
            return
        except AttributeError:
            pass

        if not self._inited:
            super().__setattr__(key, value)
        else:
            self._setattr(key, value)

    def to_dict(self):
        d = dict()
        for k, v in self.items():
            if isinstance(v, AttributeDict):
                d.update({k + '.' + sub_k: sub_v for sub_k, sub_v in v.to_dict().items()})
            elif isinstance(v[0], Redirection):
                continue
            else:
                d[k] = v[0]
        return d


class Config:
    def __init__(self, config=None):
        self._config = config or AttributeDict()
        self._serialize_options = []

    def __dir__(self):
        return list(self._config.keys())

    def __getattr__(self, item):
        config = object.__getattribute__(self, '_config')
        return getattr(config, item)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return
        setattr(self._config, key, value)

    def register_option(self, option, value, validator=None, serialize=False):
        splits = option.split('.')
        conf = self._config
        if isinstance(validator, tuple):
            validator = any_validator(*validator)

        for name in splits[:-1]:
            config = conf.get(name)
            if config is None:
                val = AttributeDict(_parent=conf)
                conf[name] = val
                conf = val
            elif not isinstance(config, dict):
                raise AttributeError(
                    f'Fail to set option: {option}, conflict has encountered')
            else:
                conf = config

        key = splits[-1]
        if conf.get(key) is not None:
            raise AttributeError(
                f'Fail to set option: {option}, option has been set')

        conf.register(key, value, validator)
        if serialize:
            self._serialize_options.append(option)

    def redirect_option(self, option, target, warn=_DEFAULT_REDIRECT_WARN):
        redir = Redirection(target, warn=warn.format(source=option, target=target))
        self.register_option(option, redir)

    def unregister_option(self, option):
        splits = option.split('.')
        conf = self._config
        for name in splits[:-1]:
            config = conf.get(name)
            if not isinstance(config, dict):
                raise AttributeError(
                    f'Fail to unregister option: {option}, conflict has encountered')
            else:
                conf = config

        key = splits[-1]
        if key not in conf:
            raise AttributeError(f'Option {option} not configured, thus failed to unregister.')
        conf.unregister(key)

    def copy(self):
        new_options = Config(deepcopy(self._config))
        return new_options

    def update(self, new_config: Union["Config", Dict]):
        if not isinstance(new_config, dict):
            new_config = new_config._config
        for option, value in new_config.items():
            try:
                self.register_option(option, value)
            except AttributeError:
                setattr(self, option, value)

    def get_serializable(self):
        d = dict()
        for k in self._serialize_options:
            parts = k.split('.')
            v = self
            for p in parts:
                v = getattr(v, p)
            d[k] = v
        return d

    def fill_serialized(self, d):
        for k, v in d.items():
            parts = k.split('.')
            cf = self
            for p in parts[:-1]:
                cf = getattr(cf, p)
            setattr(cf, parts[-1], v)

    def to_dict(self):
        return self._config.to_dict()


@contextlib.contextmanager
def option_context(config=None):
    global_options = get_global_option()

    try:
        config = config or dict()
        local_options = Config(deepcopy(global_options._config))
        local_options.update(config)
        _options_local.default_options = local_options
        yield local_options
    finally:
        _options_local.default_options = global_options


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# validators
def any_validator(*validators):
    def validate(x):
        return any(validator(x) for validator in validators)
    return validate


def all_validator(*validators):
    def validate(x):
        return all(validator(x) for validator in validators)
    return validate


def _instance_check(typ, v):
    return isinstance(v, typ)


is_null = functools.partial(operator.is_, None)
is_bool = functools.partial(_instance_check, bool)
is_integer = functools.partial(_instance_check, int)
is_float = functools.partial(_instance_check, float)
is_numeric = functools.partial(_instance_check, (float, int))
is_string = functools.partial(_instance_check, str)
is_dict = functools.partial(_instance_check, dict)
is_list = functools.partial(_instance_check, list)


def is_in(vals):
    def validate(x):
        return x in vals
    return validate


default_options = Config()
default_options.register_option('tcp_timeout', 30, validator=is_integer)
default_options.register_option('verbose', False, validator=is_bool)
default_options.register_option('kv_store', ':inproc:', validator=is_string)
default_options.register_option('check_interval', 20, validator=is_integer)
default_options.register_option('show_progress', 'auto', validator=any_validator(is_bool, is_string))
default_options.register_option('serialize_method', 'pickle')

# dataframe-related options
default_options.register_option('dataframe.mode.use_inf_as_na', False, validator=is_bool)
default_options.register_option('dataframe.use_arrow_dtype', None, validator=any_validator(is_null, is_bool))
default_options.register_option('dataframe.arrow_array.pandas_only', None, validator=any_validator(is_null, is_bool))

# learn options
assume_finite = os.environ.get('SKLEARN_ASSUME_FINITE')
if assume_finite is not None:
    assume_finite = bool(assume_finite)
working_memory = os.environ.get('SKLEARN_WORKING_MEMORY')
if working_memory is not None:
    working_memory = int(working_memory)
default_options.register_option('learn.assume_finite', assume_finite, validator=any_validator(is_null, is_bool))
default_options.register_option('learn.working_memory', working_memory, validator=any_validator(is_null, is_integer))

# the number of combined chunks in tree reduction or tree add
default_options.register_option('combine_size', 4, validator=is_integer, serialize=True)

# the default chunk store size
default_options.register_option('chunk_store_limit', 128 * 1024 ** 2, validator=is_numeric)
default_options.register_option('chunk_size', None, validator=any_validator(is_null, is_integer), serialize=True)

# rechunk
default_options.register_option('rechunk.threshold', 4, validator=is_integer, serialize=True)
default_options.register_option('rechunk.chunk_size_limit', int(1e8), validator=is_integer, serialize=True)

# deploy
default_options.register_option('deploy.open_browser', True, validator=is_bool)

# Scheduler
default_options.register_option('scheduler.assign_chunk_workers', False, validator=is_bool, serialize=True)
default_options.register_option('scheduler.enable_active_push', True, validator=is_bool, serialize=True)
default_options.register_option('scheduler.enable_chunk_relocation', False, validator=is_bool, serialize=True)
default_options.register_option('scheduler.check_interval', 1, validator=is_integer, serialize=True)
default_options.register_option('scheduler.default_cpu_usage', 1, validator=(is_integer, is_float), serialize=True)
default_options.register_option('scheduler.default_cuda_usage', 0.5, validator=(is_integer, is_float), serialize=True)
default_options.register_option('scheduler.assign_timeout', 600, validator=is_integer, serialize=True)
default_options.register_option('scheduler.execution_timeout', 600, validator=is_integer, serialize=True)
default_options.register_option('scheduler.retry_num', 4, validator=is_integer, serialize=True)
default_options.register_option('scheduler.fetch_limit', 10 * 1024 ** 2, validator=is_integer, serialize=True)
default_options.register_option('scheduler.retry_delay', 60, validator=is_integer, serialize=True)

default_options.register_option('scheduler.dump_graph_data', False, validator=is_bool)

default_options.register_option('scheduler.enable_failover', True, validator=is_bool, serialize=True)
default_options.register_option('scheduler.status_timeout', 30, validator=is_numeric, serialize=True)
default_options.register_option('scheduler.worker_blacklist_time', 3600, validator=is_numeric, serialize=True)

# enqueue operands in a batch when creating OperandActors
default_options.register_option('scheduler.batch_enqueue_initials', True, validator=is_bool, serialize=True)
# invoke assigning when where there is no ready descendants
default_options.register_option('scheduler.aggressive_assign', False, validator=is_bool, serialize=True)

# Worker
default_options.register_option('worker.spill_directory', None, validator=(is_null, is_string, is_list))
default_options.register_option('worker.disk_compression', 'lz4', validator=is_string, serialize=True)
default_options.register_option('worker.min_spill_size', '5%', validator=(is_string, is_integer))
default_options.register_option('worker.max_spill_size', '95%', validator=(is_string, is_integer))
default_options.register_option('worker.min_cache_mem_size', None, validator=(is_null, is_string, is_integer))
default_options.register_option('worker.callback_preserve_time', 3600 * 24, validator=is_integer)
default_options.register_option('worker.event_preserve_time', 3600 * 24, validator=(is_integer, is_float))
default_options.register_option('worker.copy_block_size', 64 * 1024, validator=is_integer)
default_options.register_option('worker.cuda_thread_num', 2, validator=is_integer)
default_options.register_option('worker.transfer_block_size', 1 * 1024 ** 2, validator=is_integer)
default_options.register_option('worker.transfer_compression', 'lz4', validator=is_string, serialize=True)
default_options.register_option('worker.prepare_data_timeout', 1000, validator=is_integer)
default_options.register_option('worker.peer_blacklist_time', 3600, validator=is_numeric, serialize=True)
default_options.register_option('worker.io_parallel_num', 1, validator=is_integer, serialize=True)
default_options.register_option('worker.recover_dead_process', True, validator=is_bool, serialize=True)
default_options.register_option('worker.write_shuffle_to_disk', False, validator=is_bool, serialize=True)

default_options.register_option('worker.filemerger.enabled', True, validator=is_bool, serialize=True)
default_options.register_option('worker.filemerger.concurrency', 128, validator=is_integer, serialize=True)
default_options.register_option('worker.filemerger.max_accept_size', 128 * 1024, validator=is_integer, serialize=True)
default_options.register_option('worker.filemerger.max_file_size', 32 * 1024 ** 2, validator=is_integer, serialize=True)

default_options.register_option('worker.plasma_dir', None, validator=(is_string, is_null))
default_options.register_option('worker.plasma_limit', None, validator=(is_string, is_integer, is_null))

default_options.register_option('worker.plasma_socket', '/tmp/plasma', validator=is_string)

# optimization
default_options.register_option('optimize.min_stats_count', 10, validator=is_integer)
default_options.register_option('optimize.stats_sufficient_ratio', 0.9, validator=is_float, serialize=True)
default_options.register_option('optimize.default_disk_io_speed', 10 * 1024 ** 2, validator=is_integer)
default_options.register_option('optimize.head_optimize_threshold', 1000, validator=is_integer)

default_options.register_option('optimize_tileable_graph', True, validator=is_bool)

# eager mode
default_options.register_option('eager_mode', False, validator=is_bool)

# client serialize type
default_options.register_option('client.serial_type', 'arrow', validator=is_string)

# custom log dir
default_options.register_option('custom_log_dir', None, validator=any_validator(is_null, is_string))

# vineyard
default_options.register_option('vineyard.socket', os.environ.get('VINEYARD_IPC_SOCKET', None))
default_options.register_option('vineyard.enabled', os.environ.get('WITH_VINEYARD', None) is not None)

_options_local = threading.local()
_options_local.default_options = default_options


def get_global_option():
    ret = getattr(_options_local, 'default_options', None)
    if ret is None:
        ret = _options_local.default_options = Config(deepcopy(default_options._config))

    return ret


class OptionsProxy:
    def __dir__(self):
        return dir(get_global_option())

    def __getattribute__(self, attr):
        return getattr(get_global_option(), attr)

    def __setattr__(self, key, value):
        setattr(get_global_option(), key, value)


options = OptionsProxy()

options.redirect_option('tensor.chunk_store_limit', 'chunk_store_limit')
options.redirect_option('tensor.chunk_size', 'chunk_size')
options.redirect_option('tensor.rechunk.threshold', 'rechunk.threshold')
options.redirect_option('tensor.rechunk.chunk_size_limit', 'rechunk.chunk_size_limit')
