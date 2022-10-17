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

import subprocess
import sys
import tempfile
import threading

import pytest

from .test_ray_cluster_standalone import new_ray_session_test
from ....tests.core import require_ray
from ....utils import lazy_import

ray = lazy_import("ray")


@require_ray
@pytest.mark.parametrize(
    "backend",
    [
        "mars",
        "ray",
    ],
)
def test_ray_client(backend):
    server_code = """import time
import ray.util.client.server.server as ray_client_server

server = ray_client_server.init_and_serve("{address}", num_cpus=20)
print("OK", flush=True)
while True:
    time.sleep(1)
"""

    address = "127.0.0.1:50051"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(server_code.format(address=address))
        f.flush()

        proc = subprocess.Popen([sys.executable, "-u", f.name], stdout=subprocess.PIPE)

        try:

            def _check_ready(expect_exit=False):
                while True:
                    line = proc.stdout.readline()
                    if proc.returncode is not None:
                        if expect_exit:
                            break
                        raise Exception(
                            f"Failed to start ray server at {address}, "
                            f"the return code is {proc.returncode}."
                        )
                    if b"OK" in line:
                        break

            # Avoid ray.init timeout.
            _check_ready()

            # Avoid blocking the subprocess when the stdout pipe is full.
            t = threading.Thread(target=_check_ready, args=(True,), daemon=True)
            t.start()
            try:
                import ray

                ray.client(address).connect()  # Ray 1.4
            except Exception:
                try:
                    from ray.util.client import ray

                    ray.connect(address)  # Ray 1.2
                except Exception:
                    import ray

                    ray.init(f"ray://{address}")  # Ray latest
            ray._inside_client_test = True
            try:
                new_ray_session_test(backend=backend)
            finally:
                ray._inside_client_test = False
                ray.shutdown()
        finally:
            proc.kill()
            proc.wait()
