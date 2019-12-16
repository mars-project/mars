# -*- coding: utf-8 -*-
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

import os
import sys

from ...actors import ActorNotExist, new_client
from .core import ReadinessActor


def main():
    """
    Readiness probe for Mars schedulers and workers
    """
    client = new_client()
    try:
        ref = client.actor_ref(ReadinessActor.default_uid(),
                               address='127.0.0.1:%s' % os.environ['MARS_K8S_SERVICE_PORT'])
        sys.exit(0 if client.has_actor(ref) else 1)
    except (ActorNotExist, ConnectionRefusedError) as ex:  # noqa: E722
        sys.stderr.write('Probe error: %s' % type(ex).__name__)
        sys.exit(1)


if __name__ == '__main__':   # pragma: no branch
    main()
