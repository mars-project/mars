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

from abc import ABC
from typing import List, Tuple, Dict


class AbstractSchedulingAPI(ABC):
    async def submit_subtasks(self, subtasks: List, priorities: List[Tuple]):
        """
        Submit subtasks into scheduling service

        Parameters
        ----------
        subtasks
            list of subtasks to be submitted to service
        priorities
            list of priorities of subtasks
        """

    async def update_subtask_priorities(self, task_to_priorities: Dict[str, Tuple]):
        """
        Update priorities of subtasks

        Parameters
        ----------
        task_to_priorities
            mapping from subtask ids to priorities
        """

    async def cancel_subtasks(self, subtask_ids: List[str]):
        """
        Cancel pending and running subtasks.

        Parameters
        ----------
        subtask_ids
            ids of subtasks to cancel
        """

    async def finish_subtasks(self, subtask_ids: List[str]):
        """
        Mark subtasks as finished, letting scheduling service to schedule
        next tasks in the ready queue

        Parameters
        ----------
        subtask_ids
            ids of subtasks to mark as finished
        """

    async def get_subtask_details(self, subtask_ids: List[str]) -> Dict[str, Dict]:
        """
        Get details of subtasks

        Parameters
        ----------
        subtask_ids
            ids of subtasks.

        Returns
        -------
        out
            mapping from subtask ids to details.
        """

    async def wait_subtasks(self, subtask_ids: List[str]):
        """
        Wait subtasks till all tasks finished

        Parameters
        ----------
        subtask_ids
            ids of subtasks to wait for.
        """
