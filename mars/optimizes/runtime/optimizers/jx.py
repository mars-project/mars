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
from .base_optimizer import Optimizer


class JaxOptimizer(Optimizer):
    def __init__(self, graph):
        super(JaxOptimizer, self).__init__(graph)

    def optimize(self, keys=None):
        self.compose(keys)

    def _can_break(self, node):
        if not self._support(node):
            return True
        if self.graph.count_successors(node) != 1:
            return True
        return False

    def _support(self, node):
        op = node.op
        if hasattr(op, 'jax_function') and op.jax_function() is not NotImplementedError:
            return True
        return False

    def _can_skip(self, node):
        if super(JaxOptimizer, self)._can_skip(node):
            return True
        return False

    @staticmethod
    def _get_fused_chunk(tail_node):
        from ....tensor.fuse import TensorJaxFuseChunk
        return TensorJaxFuseChunk(dtype=tail_node.dtype)
