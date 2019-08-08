#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ....tensor.fuse.ne import NUMEXPR_INSTALLED
from ....tensor.fuse.jax import JAX_INSTALLED
from .ne import NeOptimizer
from .cp import CpOptimizer
from .jx import JaxOptimizer


class Optimizer(object):
    engine_dic = {'numexpr': NeOptimizer,
                  'cupy': CpOptimizer, 'jax': JaxOptimizer}

    def __init__(self, graph, engines=None):
        self._graph = graph
        self._engines = []
        if not engines:
            # the sequence of optimize
            if JAX_INSTALLED:
                self._engines.append('jax')
            if NUMEXPR_INSTALLED:
                self._engines.append('numexpr')

        else:
            self._engines = engines
            # just one selected engine
            if isinstance(self._engines, str):
                self._engines = [self._engines]

            # filter numpy
            self._engines = [e for e in self._engines if e != 'numpy']

    def optimize(self, keys=None):
        self._graph.decompose()
        # no optimization, only numpy
        if len(self._engines) == 0:
            return
        for engine in self._engines:
            optimizer = self.engine_dic[engine](self._graph)
            optimizer.optimize(keys=keys)
