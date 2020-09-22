# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import base64
import json
import time
import logging
import pickle
import sys
import uuid
from io import BytesIO
from numbers import Integral

import numpy as np

from ..config import options
from ..errors import ResponseMalformed, ExecutionInterrupted, ExecutionFailed, \
    ExecutionStateUnknown, ExecutionNotStopped
from ..operands import Fetch
from ..serialize import dataserializer
from ..serialize.dataserializer import pyarrow
from ..tensor.core import Indexes
from ..utils import build_tileable_graph, sort_dataframe_result, numpy_dtype_from_descr_json

logger = logging.getLogger(__name__)


class Session(object):
    def __init__(self, endpoint, session_id=None, req_session=None, verify_ssl=True,
                 **session_kwargs):
        self._endpoint = endpoint.rstrip('/')
        self._session_id = session_id
        self._session_kwargs = session_kwargs
        # dict structure: {tileable_key -> graph_key, tileable_ids}
        # dict value is a tuple object which records graph key and tileable id
        self._executed_tileables = dict()

        self._serial_type = None
        self._pickle_protocol = pickle.HIGHEST_PROTOCOL

        if req_session:
            self._req_session = req_session
        else:
            import requests
            from requests.adapters import HTTPAdapter

            self._req_session = requests.Session()
            self._req_session.mount('http://stackoverflow.com', HTTPAdapter(max_retries=5))

        self._req_session.verify = verify_ssl
        if not verify_ssl:
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:  # pragma: no cover
                pass

        self._main()

    @property
    def session_id(self):
        return self._session_id

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, url):
        self._endpoint = url

    def _main(self):
        if pyarrow is None:
            self._serial_type = dataserializer.SerialType.PICKLE
        else:
            self._serial_type = dataserializer.SerialType(options.client.serial_type.lower())

        session_kw = self._session_kwargs.copy()
        session_kw['pyver'] = '.'.join(str(v) for v in sys.version_info[:3])
        session_kw['pickle_protocol'] = self._pickle_protocol
        if pyarrow is not None:
            session_kw['arrow_version'] = pyarrow.__version__

        if self._session_id is None:
            resp = self._req_session.post(self._endpoint + '/api/session', data=session_kw)

            if resp.status_code >= 400:
                raise SystemError('Failed to create mars session: ' + resp.reason)
        else:
            resp = self._req_session.get(
                self._endpoint + '/api/session/' + self._session_id, params=session_kw)
            if resp.status_code == 404:
                raise ValueError(f'The session with id = {self._session_id} doesn\'t exist')
            if resp.status_code >= 400:
                raise SystemError('Failed to check mars session.')

        content = json.loads(resp.text)
        self._session_id = content['session_id']
        self._pickle_protocol = content.get('pickle_protocol', pickle.HIGHEST_PROTOCOL)

        # as pyarrow will use pickle.HIGHEST_PROTOCOL to pickle, we need to use
        # SerialType.PICKLE when pickle protocol between client and server
        # does not agree with each other
        if not content.get('arrow_compatible') or self._pickle_protocol != pickle.HIGHEST_PROTOCOL:
            self._serial_type = dataserializer.SerialType.PICKLE

    def _get_tileable_graph_key(self, tileable_key):
        return self._executed_tileables[tileable_key][0]

    def _set_tileable_graph_key(self, tileable, graph_key):
        tileable_key = tileable.key
        tileable_id = tileable.id
        if tileable_key in self._executed_tileables:
            self._executed_tileables[tileable_key][1].add(tileable_id)
        else:
            self._executed_tileables[tileable_key] = graph_key, {tileable_id}

    @staticmethod
    def _handle_json_response(resp, allow_empty=True, raises=True):
        try:
            resp_txt = resp.text
            if allow_empty:
                resp_txt = resp_txt or '{}'
            resp_json = json.loads(resp_txt)
        except json.JSONDecodeError:
            text_part = resp.text if len(resp.text) < 100 else resp.text[:100] + '...'
            raise ResponseMalformed(f'Failed to parse server response. Status={resp.status_code} '
                                    f'Response="{text_part}"')

        if raises and resp.status_code >= 400:
            exc_info = pickle.loads(base64.b64decode(resp_json['exc_info']))
            raise exc_info[1].with_traceback(exc_info[2])
        return resp_json

    def _check_response_finished(self, graph_url, timeout=None):
        import requests
        try:
            resp = self._req_session.get(graph_url, params={'wait_timeout': timeout})
        except requests.ConnectionError as ex:
            err_msg = str(ex)
            if 'ConnectionResetError' in err_msg or 'Connection refused' in err_msg:
                return False
            raise

        if resp.status_code == 504:
            logging.debug('Gateway Time-out, try again')
            return False
        if resp.status_code >= 400:
            raise SystemError(f'Failed to obtain execution status. Code: {resp.status_code}, '
                              f'Reason: {resp.reason}, Content:\n{resp.text}')

        resp_json = self._handle_json_response(resp, raises=False)
        if resp_json['state'] == 'succeeded':
            return True
        elif resp_json['state'] in ('running', 'preparing'):
            return False
        elif resp_json['state'] in ('cancelled', 'cancelling'):
            raise ExecutionInterrupted
        elif resp_json['state'] == 'failed':
            if 'exc_info' in resp_json:
                exc_info = pickle.loads(base64.b64decode(resp_json['exc_info']))
                exc = exc_info[1].with_traceback(exc_info[2])
                raise ExecutionFailed('Graph execution failed.') from exc
            else:
                raise ExecutionFailed('Graph execution failed with unknown reason.')
        raise ExecutionStateUnknown('Unknown graph execution state ' + resp_json['state'])

    def run(self, *tileables, **kw):
        timeout = kw.pop('timeout', -1)
        compose = kw.pop('compose', True)
        fetch = kw.pop('fetch', True)
        name = kw.pop('name', None)
        if kw:
            raise TypeError(f'run got unexpected key arguments {kw!r}')

        # those executed tileables should fetch data directly, submit the others
        run_tileables = [t for t in tileables if t.key not in self._executed_tileables]

        if name is not None:
            if not isinstance(name, (list, tuple)):
                name = [name]
            if len(name) != len(tileables):
                raise TypeError('Name must match execute tileables')
            name = ','.join(name)

        graph = build_tileable_graph(run_tileables, set(self._executed_tileables.keys()))
        targets = [t.key for t in run_tileables]

        targets_join = ','.join(targets)
        session_url = self._endpoint + '/api/session/' + self._session_id
        graph_json = graph.to_json(data_serial_type=self._serial_type, pickle_protocol=self._pickle_protocol)

        resp_json = self._submit_graph(graph_json, targets_join, names=name or '', compose=compose)
        graph_key = resp_json['graph_key']
        graph_url = f'{session_url}/graph/{graph_key}'

        exec_start_time = time.time()
        time_elapsed = 0
        check_interval = options.check_interval
        while timeout <= 0 or time_elapsed < timeout:
            timeout_val = min(check_interval, timeout - time_elapsed) if timeout > 0 else check_interval
            try:
                if self._check_response_finished(graph_url, timeout_val):
                    break
            except KeyboardInterrupt:
                resp = self._req_session.delete(graph_url)
                if resp.status_code >= 400:
                    raise ExecutionNotStopped(
                        f'Failed to stop graph execution. Code: {resp.status_code}, '
                        f'Reason: {resp.reason}, Content:\n{resp.text}')
            finally:
                time_elapsed = time.time() - exec_start_time

        if 0 < timeout < time.time() - exec_start_time:
            raise TimeoutError

        for t in tileables:
            self._set_tileable_graph_key(t, graph_key)

        if not fetch:
            return
        else:
            return self.fetch(*tileables)

    def _is_executed(self, tileable):
        # if tileble.key in executed tileables
        # or it's a fetch already
        return tileable.key in self._executed_tileables or \
               isinstance(tileable.op, Fetch)

    def fetch(self, *tileables, **kw):
        from ..tensor.indexing import TensorIndex
        from ..dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        timeout = kw.pop('timeout', None)
        if kw:
            raise TypeError(f'fetch got unexpected key arguments {kw!r}')

        results = list()
        for tileable in tileables:
            if tileable.key not in self._executed_tileables and \
                    isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem)):
                to_fetch_tileable = tileable.inputs[0]
                indexes = tileable.op.indexes
                if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                    raise ValueError('Only support fetch data slices')
            else:
                to_fetch_tileable = tileable
                indexes = []

            if not self._is_executed(to_fetch_tileable):
                raise ValueError('Cannot fetch the unexecuted tileable')

            key = to_fetch_tileable.key
            indexes_str = json.dumps(Indexes(indexes).to_json(), separators=(',', ':'))

            session_url = f'{self._endpoint}/api/session/{self._session_id}'
            compression_str = ','.join(v.value for v in dataserializer.get_supported_compressions())
            params = dict(compressions=compression_str, slices=indexes_str,
                          serial_type=self._serial_type.value, pickle_protocol=self._pickle_protocol)
            data_url = f'{session_url}/graph/{self._get_tileable_graph_key(key)}/data/{key}'
            resp = self._req_session.get(data_url, params=params, timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f'Failed to fetch data from server. Code: {resp.status_code}, '
                                 f'Reason: {resp.reason}, Content:\n{resp.text}')
            result_data = dataserializer.loads(resp.content)
            results.append(sort_dataframe_result(tileable, result_data))
        return results

    @classmethod
    def _process_int_or_dict_argument(cls, argument, name, params):
        if argument is None:
            return
        if not isinstance(argument, dict):
            params[name] = argument
        else:
            params[name] = ','.join(f'{k}={v}' for k, v in argument.items())

    def fetch_tileable_op_logs(self, tileable_op_key, offsets=None, sizes=None):
        url = f'{self._endpoint}/api/session/{self._session_id}/op/{tileable_op_key}/log'
        params = dict()
        self._process_int_or_dict_argument(offsets, 'offsets', params)
        self._process_int_or_dict_argument(sizes, 'sizes', params)
        resp = self._req_session.get(url, params=params)
        if resp.status_code >= 400:
            raise ValueError(f'Failed to fetch log from server. Code: {resp.status_code}, '
                             f'Reason: {resp.reason}, Content:\n{resp.text}')
        return json.loads(resp.content)

    def fetch_log(self, tileables, offsets=None, sizes=None):
        from ..custom_log import fetch

        return fetch(tileables, self, offsets=offsets, sizes=sizes)

    def get_named_tileable_infos(self, name):
        from ..context import TileableInfos

        url = f'{self._endpoint}/api/session/{self._session_id}'
        params = dict(name=name)
        resp = self._req_session.get(url, params=params)
        if resp.status_code >= 400:  # pragma: no cover
            raise ValueError(f'Failed to get tileable key from server. Code: {resp.status_code}, '
                             f'Reason: {resp.reason}, Content:\n{resp.text}')
        tileable_key = self._handle_json_response(resp)['tileable_key']
        nsplits = self._get_tileable_nsplits(tileable_key)
        shape = tuple(sum(s) for s in nsplits)
        return TileableInfos(tileable_key, shape)

    def create_mutable_tensor(self, name, shape, dtype, fill_value=None, chunk_size=None, *_, **__):
        from ..tensor.utils import create_mutable_tensor
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        tensor_url = f'{session_url}/mutable-tensor/{name}?action=create'
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        # avoid built-in scalar dtypes are made into one-field record type.
        if dtype.fields:
            dtype_descr = dtype.descr
        else:
            dtype_descr = str(dtype)
        tensor_json = {
            'shape': shape,
            'dtype': dtype_descr,
            'fill_value': fill_value,
            'chunk_size': chunk_size,
        }
        resp = self._req_session.post(tensor_url, json=tensor_json)
        shape, dtype, chunk_size, chunk_keys, chunk_eps = self._handle_json_response(resp)
        return create_mutable_tensor(name, chunk_size, shape, numpy_dtype_from_descr_json(dtype),
                                     chunk_keys, chunk_eps)

    def get_mutable_tensor(self, name):
        from ..tensor.utils import create_mutable_tensor
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        tensor_url = f'{session_url}/mutable-tensor/{name}'
        resp = self._req_session.get(tensor_url)
        shape, dtype, chunk_size, chunk_keys, chunk_eps = self._handle_json_response(resp)
        return create_mutable_tensor(name, chunk_size, shape, numpy_dtype_from_descr_json(dtype),
                                     chunk_keys, chunk_eps)

    def write_mutable_tensor(self, tensor, index, value):
        """
        How to serialize index and value:

        1. process_index and serialize it as json
        2. the payload of POST request:

            * a int64 value indicate the size of index json
            * ascii-encoded bytes of index json
            * pyarrow serialized bytes of `value`
        """
        from ..tensor.core import Indexes
        from ..serialize import dataserializer

        index = Indexes(_indexes=index)
        index_bytes = json.dumps(index.to_json()).encode('ascii')
        bio = BytesIO()
        bio.write(np.int64(len(index_bytes)).tobytes())
        bio.write(index_bytes)
        dataserializer.dump(value, bio)

        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        tensor_url = f'{session_url}/mutable-tensor/{tensor.name}'
        resp = self._req_session.put(tensor_url, data=bio.getvalue(),
                                     headers={'Content-Type': 'application/octet-stream'})
        self._handle_json_response(resp)

    def seal(self, tensor):
        from ..tensor.utils import create_fetch_tensor
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        tensor_url = f'{session_url}/mutable-tensor/{tensor.name}?action=seal'
        resp = self._req_session.post(tensor_url)
        graph_key_hex, tileable_key, tensor_id, tensor_meta = self._handle_json_response(resp)
        self._executed_tileables[tileable_key] = uuid.UUID(graph_key_hex), {tensor_id}

        # # Construct Tensor on the fly.
        shape, dtype, chunk_size, chunk_keys, _ = tensor_meta
        return create_fetch_tensor(chunk_size, shape, numpy_dtype_from_descr_json(dtype),
                                   tensor_key=tileable_key, chunk_keys=chunk_keys)

    def _get_tileable_nsplits(self, tileable_key):
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        graph_key = self._get_tileable_graph_key(tileable_key)
        url = f'{session_url}/graph/{graph_key}/data/{tileable_key}?type=nsplits'
        resp = self._req_session.get(url)
        new_nsplits = self._handle_json_response(resp)
        return new_nsplits

    def _update_tileable_shape(self, tileable):
        tileable_key = tileable.key
        new_nsplits = self._get_tileable_nsplits(tileable_key)
        tileable._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tileable.nsplits = new_nsplits

    def decref(self, *keys):
        for tileable_key, tileable_id in keys:
            if tileable_key not in self._executed_tileables:
                continue
            graph_key, ids = self._executed_tileables[tileable_key]

            if tileable_id in ids:
                ids.remove(tileable_id)
                # for those same key tileables, do decref only when all those tileables are garbage collected
                if len(ids) != 0:
                    continue
                self.delete_data(tileable_key)

    def delete_data(self, tileable_key, wait=False):
        if tileable_key not in self._executed_tileables:
            return
        graph_key, _ids = self._executed_tileables[tileable_key]
        data_url = f'{self._endpoint}/api/session/{self._session_id}/graph/{graph_key}' \
                   f'/data/{tileable_key}?wait={1 if wait else 0}'
        self._req_session.delete(data_url)
        self._executed_tileables.pop(tileable_key, None)

    def stop(self, graph_key):
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        graph_url = session_url + '/graph/' + graph_key
        resp = self._req_session.delete(graph_url)
        if resp.status_code >= 400:
            raise SystemError(f'Failed to stop graph execution. Code: {resp.status_code}, '
                              f'Reason: {resp.reason}, Content:\n{resp.text}')

    def _submit_graph(self, graph_json, targets, names=None, compose=True):
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        resp = self._req_session.post(session_url + '/graph', dict(
            graph=json.dumps(graph_json),
            target=targets,
            names=names,
            compose='1' if compose else '0'
        ))
        return self._handle_json_response(resp)

    def get_graph_states(self):
        resp = self._req_session.get(f'{self._endpoint}/api/session/{self._session_id}/graph')
        return self._handle_json_response(resp)

    def close(self):
        session_url = f'{self._endpoint}/api/session/{self._session_id}'
        for key in list(self._executed_tileables.keys()):
            self.delete_data(key, wait=True)
        resp = self._req_session.delete(session_url)
        if resp.status_code >= 400:
            raise SystemError('Failed to close mars session.')

    def check_service_ready(self, timeout=1):
        import requests
        try:
            resp = self._req_session.get(self._endpoint + '/api', timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            return False
        if resp.status_code >= 400:
            return False
        return True

    def count_workers(self):
        resp = self._req_session.get(self._endpoint + '/api/worker?action=count', timeout=1)
        return self._handle_json_response(resp)

    def rescale_workers(self, new_scale, min_workers=None, wait=True, timeout=None):
        data = json.dumps(dict(new_scale=new_scale, min_workers=min_workers))
        wait_req = 1 if wait else 0
        resp = self._req_session.patch(f'{self._endpoint}/api/worker?action=count&wait={wait_req}',
                                       data, timeout=timeout)
        return self._handle_json_response(resp)

    def get_workers_meta(self):
        resp = self._req_session.get(self._endpoint + '/api/worker', timeout=1)
        return self._handle_json_response(resp)

    def get_task_count(self):
        resp = self._req_session.get(f'{self._endpoint}/api/session/{self._session_id}/graph')
        return len(self._handle_json_response(resp))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
