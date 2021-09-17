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

from typing import List, Union
from .... import oscar as mo
from ..utils import normailize_index
from ...cluster.api import ClusterAPI


class MutableTensor:
    def __init__(self,
                 ref: mo.ActorRef):
        self._ref = ref

    async def __getitem__(self, index: Union[int, List[int]]):
        '''
        Function
        ----------
        read a single point of the tensor.

        Parameters:
        ----------
        index: List[int]
        If there is need to read n points of the tensor, suppose the dimension of the tensor is m.\n
        for the i_th point, it would be represented by (a[i][0],a[i][1]....a[i][m-1]),i in range(0,n)\n
        then we extract a[i][j] with the same j and put them together.\n
        It's the way to get the index for the parameter, its form is like \n
        [(a[0][0],a[1][0]...a[n-1][0]),(a[0][1],a[1][1]...a[n-1][1])...(a[0][m-1],a[1][m-1]...a[n-1][m-1])]\n
        e.g.
        Assumed the shape of index is (100,200,300) and we want to read the (0,0,0) (10,20,30) (40,50,80)\n
        the index should be ((0,10,40),(0,20,50),(0,30,80))

        Returns
        -------
        the value of the points
        '''
        index = normailize_index(index)
        result = await self._ref.read(index)
        return result

    async def write(self, index, value):
        '''
        Function
        ----------
        read a single point of the tensor.

        Parameters:
        ----------
        index: List[int]
        If there is need to read n points of the tensor, suppose the dimension of the tensor is m.\n
        for the i_th point, it would be represented by (a[i][0],a[i][1]....a[i][m-1]),i in range(0,n)\n
        then we extract a[i][j] with the same j and put them together.\n
        It's the way to get the index for the parameter, its form is like \n
        [(a[0][0],a[1][0]...a[n-1][0]),(a[0][1],a[1][1]...a[n-1][1])...(a[0][m-1],a[1][m-1]...a[n-1][m-1])]\n
        e.g.
        Assumed the shape of index is (100,200,300) and we want to read the (0,0,0) (10,20,30) (40,50,80)\n
        the index should be ((0,10,40),(0,20,50),(0,30,80))
        '''
        await self._ref.write(index, value)
