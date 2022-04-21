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

import logging
import os

from ... import oscar as mo
from ...services import NodeRole
from ...utils import get_next_port
from .cmdline import OscarCommandRunner
from .local import start_supervisor, stop_supervisor
from .pool import create_supervisor_actor_pool


logger = logging.getLogger(__name__)


class SupervisorCommandRunner(OscarCommandRunner):
    command_description = "Mars Supervisor"
    node_role = NodeRole.SUPERVISOR

    def __init__(self):
        super().__init__()
        self._endpoint_file_name = None

    def config_args(self, parser):
        super().config_args(parser)
        parser.add_argument("-w", "--web-port", help="web port of the service")
        parser.add_argument(
            "--n-process", help="number of supervisor processes", default="1"
        )

    def parse_args(self, parser, argv, environ=None):
        args = super().parse_args(parser, argv, environ=environ)

        if args.endpoint is None:
            args.endpoint = f"{args.host}:{get_next_port()}"
        self._endpoint_file_name = self._write_supervisor_endpoint_file(args)

        args.supervisors = f"{args.supervisors},{args.endpoint}".strip(",")

        web_config = self.config.get("web", {})
        if args.web_port is not None:
            web_config["host"] = args.endpoint.split(":", 1)[0]
            web_config["port"] = int(args.web_port)
        self.config["web"] = web_config

        return args

    async def create_actor_pool(self):
        return await create_supervisor_actor_pool(
            self.args.endpoint,
            n_process=int(self.args.n_process),
            ports=self.ports,
            modules=self.args.load_modules,
            logging_conf=self.logging_conf,
            subprocess_start_method="forkserver" if os.name == "nt" else "spawn",
            metrics=self.config.get("metrics", {}),
        )

    async def start_services(self):
        start_web = await start_supervisor(
            self.pool.external_address,
            self.args.supervisors,
            self.args.load_modules,
            self.config,
        )
        if start_web:
            from ...services.web.supervisor import WebActor

            web_actor = await mo.actor_ref(
                WebActor.default_uid(), address=self.pool.external_address
            )
            web_address = await web_actor.get_web_address()
        else:  # pragma: no cover
            web_address = "<web not started>"
        logger.warning(
            "Supervisor started at %s, web address: %s",
            self.pool.external_address,
            web_address,
        )

    async def stop_services(self):
        if self._endpoint_file_name is not None:  # pragma: no branch
            try:
                os.unlink(self._endpoint_file_name)
            except OSError:  # pragma: no cover
                pass
        return await stop_supervisor(self.pool.external_address, self.config)


main = SupervisorCommandRunner()

if __name__ == "__main__":  # pragma: no branch
    main()
