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

import itertools
from functools import partial
from typing import Callable, Dict

import numpy as np

from .....core import (
    TileableType,
    ChunkGraph,
    OBJECT_TYPE,
    enter_mode,
    register,
    unregister,
)
from .....core.operand import Fetch
from .....resource import Resource
from .....tests.core import _check_args, ObjectCheckMixin
from .....typing import BandType, ChunkType
from ....subtask import Subtask, SubtaskGraph
from ...analyzer import GraphAnalyzer
from ..preprocessor import CancellableTiler, TaskPreprocessor


class CheckedTaskPreprocessor(ObjectCheckMixin, TaskPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_chunk_shapes = dict()
        self._tileable_checked = dict()

        check_options = dict()
        kwargs = self._task.extra_config or dict()
        self._operand_tile_handlers = operand_tile_handlers = kwargs.pop(
            "operand_tile_handlers", dict()
        )
        for op, tile_handler in operand_tile_handlers.items():
            register(op, tile_handler)
        for key in _check_args:
            check_options[key] = kwargs.get(key, True)
        self._check_options = check_options

    def _get_done(self):
        return super()._get_done()

    def _set_done(self, is_done: bool):
        super()._set_done(is_done)
        for op in self._operand_tile_handlers:
            unregister(op)

    done = property(_get_done, _set_done)

    def _check_nsplits(self, tiled: TileableType):
        if tiled.nsplits == () and len(tiled.chunks) == 1:
            return

        nsplit_chunk_shape = tuple(len(s) for s in tiled.nsplits)
        if nsplit_chunk_shape != tiled.chunk_shape:
            raise AssertionError(
                "Operand %r: shape of nsplits %r not consistent with chunk shape %r"
                % (tiled.op, nsplit_chunk_shape, tiled.chunk_shape)
            ) from None

        nsplit_shape = tuple(np.sum(s) for s in tiled.nsplits)
        try:
            self.assert_shape_consistent(nsplit_shape, tiled.shape)
        except AssertionError:
            raise AssertionError(
                "Operand %r: shape computed from nsplits %r -> %r not consistent with real shape %r"
                % (tiled.op, tiled.nsplits, nsplit_shape, tiled.shape)
            ) from None

        for c in tiled.chunks:
            try:
                tiled_c = tiled.cix[c.index]
            except ValueError as ex:
                raise AssertionError(
                    "Operand %r: Malformed index %r, nsplits is %r. Raw error is %r"
                    % (c.op, c.index, tiled.nsplits, ex)
                ) from None

            if tiled_c is not c:
                raise AssertionError(
                    "Operand %r: Cannot spot chunk via index %r, nsplits is %r"
                    % (c.op, c.index, tiled.nsplits)
                )
        for cid, shape in enumerate(itertools.product(*tiled.nsplits)):
            chunk_shape = (
                self._raw_chunk_shapes.get(tiled.chunks[cid].key)
                or tiled.chunks[cid].shape
            )
            if len(shape) != len(chunk_shape):
                raise AssertionError(
                    "Operand %r: Shape in nsplits %r does not meet shape in chunk %r"
                    % (tiled.chunks[cid].op, shape, chunk_shape)
                )
            for s1, s2 in zip(shape, chunk_shape):
                if (not (np.isnan(s1) and np.isnan(s2))) and s1 != s2:
                    raise AssertionError(
                        "Operand %r: Shape in nsplits %r does not meet shape in chunk %r"
                        % (tiled.chunks[cid].op, shape, chunk_shape)
                    )

    def post_chunk_graph_execution(self):
        for tileable in self.tileable_graph:
            tiled_tileable = self.tile_context.get(tileable)
            if (
                tiled_tileable is not None
                and self._check_options["check_nsplits"]
                and tileable.key not in self._tileable_checked
                and not isinstance(tileable, OBJECT_TYPE)
            ):
                self._check_nsplits(tiled_tileable)
                self._tileable_checked[tileable.key] = True

    def _get_tiler_cls(self) -> Callable:
        extra_config = self._task.extra_config or dict()
        check_duplicated_submission = extra_config.get(
            "check_duplicated_submission", True
        )
        return partial(
            CancellableTiler,
            cancelled=self._cancelled,
            check_duplicated_submission=check_duplicated_submission,
        )

    @enter_mode(build=True)
    def analyze(
        self,
        chunk_graph: ChunkGraph,
        chunk_to_subtasks: Dict[ChunkType, Subtask],
        available_bands: Dict[BandType, Resource],
        stage_id: str,
        op_to_bands: Dict[str, BandType] = None,
    ) -> SubtaskGraph:
        # record shapes generated in tile
        for n in chunk_graph:
            self._raw_chunk_shapes[n.key] = getattr(n, "shape", None)
        task = self._task
        analyzer = GraphAnalyzer(
            chunk_graph, available_bands, task, self._config, chunk_to_subtasks
        )
        subtask_graph = analyzer.gen_subtask_graph()
        results = set(
            analyzer._chunk_to_copied[c]
            for c in chunk_graph.results
            if not isinstance(c.op, Fetch)
        )
        for subtask in subtask_graph:
            if subtask.extra_config is None:
                subtask.extra_config = dict()
            if all(c not in results for c in subtask.chunk_graph.results):
                subtask.extra_config["check_all"] = False
            else:
                subtask.extra_config["check_keys"] = [
                    c.key for c in subtask.chunk_graph.results if c in results
                ]
        return subtask_graph
