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

from typing import List, Dict


class ClusterApi:
    @staticmethod
    async def create(config: Dict) -> "ClusterApi":
        pass

    async def refresh_resources(self):
        """
        Refresh resource info of current node into service
        """

    async def get_supervisor(self, key: str) -> str:
        """
        Get supervisor address hosting the specified session

        Parameters
        ----------
        key : str
            key for a supervisor address

        Returns
        -------
        out : str
            address of the supervisor
        """

    async def watch_supervisors(self) -> List[str]:
        """
        Watch changes of supervisors

        Returns
        -------
        out: List[str]
            list of supervisor addresses
        """

    async def watch_workers(self) -> List[Dict[str, Dict]]:
        """
        Watch changes of workers

        Returns
        -------
        out: List[Dict[str, Dict]
            dict of worker resources by addresses and bands
        """

    async def watch_single_worker(self, worker_address: str) -> Dict:
        """
        Watch worker info update

        Parameters
        ----------
        worker_address : str
            address of worker

        Returns
        -------
        out: Dict
            info of worker
        """

    async def get_worker_info(self, worker_address: str):
        """
        Get worker info

        Parameters
        ----------
        worker_address : str
            address of worker

        Returns
        -------
        out: Dict
            info of worker
        """
