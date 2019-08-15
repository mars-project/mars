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

engine_dic = {'numexpr': NeOptimizer,
              'cupy': CpOptimizer, 'jax': JaxOptimizer}


def optimize(graph, engines, keys=None):
    if engines is None:
        # the sequence of optimize
        engines = []
        if JAX_INSTALLED:
            engines.append('jax')
        if NUMEXPR_INSTALLED:
            engines.append('numexpr')

    else:
        if not isinstance(engines, (tuple, list)):
            engines = [engines]

    # decompose the graph
    graph.decompose()
    # no optimization, only numpy
    if len(engines) == 0:
        return
    for engine in engines:
        if engine not in engine_dic:
            continue
        optimizer = engine_dic[engine](graph)
        optimizer.optimize(keys=keys)


