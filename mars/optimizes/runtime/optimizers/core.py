#!/usr/bin/env python
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

from .cp import CpRuntimeOptimizer
from .ne import NeRuntimeOptimizer
from .dataframe import DataFrameRuntimeOptimizer


class RuntimeOptimizer:
    engine_dic = {
        'numpy': None,  # DO NOT optimize
        'numexpr': NeRuntimeOptimizer,
        'cupy': CpRuntimeOptimizer,
        'dataframe': DataFrameRuntimeOptimizer,
    }

    def __init__(self, graph, engine=None):
        # conver to DAG first if graph is not a DAG
        self._dag = graph.to_dag()
        self._graph = graph
        if engine is None:
            # add default optimizers
            engine = ['numexpr', 'cupy', 'dataframe']
        elif not isinstance(engine, (list, tuple)):
            engine = [engine]
        self._engine = engine

    def optimize(self, keys=None, check_availablility=True):
        self._dag.decompose()
        for e in self._engine:
            optimizer_cls = self.engine_dic[e]
            if optimizer_cls is None:
                continue
            if check_availablility and not optimizer_cls.is_available():
                continue
            optimizer = optimizer_cls(self._dag)
            optimizer.optimize(keys=keys)
        # copy back to the original graph,
        # only when the original graph is not a DAG
        self._dag.copyto(self._graph)
