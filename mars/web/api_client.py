# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import json

import requests

from ..config import options
from .session import Session


class MarsApiClient(object):
    def __init__(self, endpoint):
        self._endpoint = endpoint
        self._req_session = requests.Session()

        from requests.adapters import HTTPAdapter
        self._req_session.mount('http://stackoverflow.com', HTTPAdapter(max_retries=5))

    @property
    def endpoint(self):
        return self._endpoint

    def create_session(self, local_options=False):
        assert self._endpoint

        args = dict()
        if local_options:
            options_dict = options.get_serializable()
            args['user_options'] = json.dumps(options_dict)

        return Session(self._endpoint, args)

    def check_service_ready(self, timeout=1):
        try:
            resp = self._req_session.get(self._endpoint + '/api', timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            return False
        if resp.status_code >= 400:
            return False
        return True

    def count_workers(self):
        resp = self._req_session.get(self._endpoint + '/api/worker', timeout=1)
        return json.loads(resp.text)


def get_client(endpoint):
    return MarsApiClient(endpoint)
