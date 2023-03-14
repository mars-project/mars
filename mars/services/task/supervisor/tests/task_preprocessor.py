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
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List

import numpy as np

from .....core import (
    TileableType,
    ChunkGraph,
    OBJECT_TYPE,
    enter_mode,
    register,
    unregister,
)
from .....core.operand import Fetch, ShuffleProxy
from .....core.operand.shuffle import ShuffleFetchType
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
        self._check_duplicated_operand_keys = bool(
            kwargs.get("check_duplicated_operand_keys")
        )

    def _get_done(self):
        return super()._get_done()

    def _set_done(self, is_done: bool):
        super()._set_done(is_done)
        for op in self._operand_tile_handlers:
            unregister(op)

    done = property(_get_done, _set_done)

    def _check_nsplits(self, tiled: TileableType):
        if tiled.nsplits is None or (tiled.nsplits == () and len(tiled.chunks) == 1):
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
        shuffle_fetch_type: ShuffleFetchType = None,
    ) -> SubtaskGraph:
        checked_chunks = set()
        for tileable in self.tileable_graph:
            try:
                tiled = self.get_tiled(tileable)
                self._check_shuffle_reduce_chunks(tiled.chunks, checked_chunks)
            except KeyError:
                pass

        # check if duplicated operand keys exist
        if self._check_duplicated_operand_keys and len(
            {c.key for c in chunk_graph}
        ) < len(
            chunk_graph
        ):  # pragma: no cover
            raise AssertionError("Duplicated operands exist")
        # record shapes generated in tile
        for n in chunk_graph:
            self._raw_chunk_shapes[n.key] = getattr(n, "shape", None)
        task = self._task
        analyzer = GraphAnalyzer(
            chunk_graph,
            available_bands,
            task,
            self._config,
            chunk_to_subtasks,
            shuffle_fetch_type=shuffle_fetch_type,
            map_reduce_id_to_infos=self.map_reduce_id_to_infos,
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
            proxy_chunks = [
                c for c in subtask.chunk_graph if isinstance(c.op, ShuffleProxy)
            ]
            if proxy_chunks:
                assert len(proxy_chunks) == 1, proxy_chunks
                proxy_chunk_key = proxy_chunks[0].key
                proxy_chunk = next(c for c in chunk_graph if c.key == proxy_chunk_key)
                reducer_chunks = chunk_graph.successors(proxy_chunk)
                n_reducers_list = [c.op.n_reducers for c in reducer_chunks]
                n_reducers = n_reducers_list[0]
                reducer_ordinals = [c.op.reducer_ordinal for c in reducer_chunks]
                assert set(reducer_ordinals).issubset(list(range(n_reducers))), (
                    reducer_ordinals,
                    n_reducers,
                )
                assert len(set(n_reducers_list)) == 1, n_reducers_list
                mapper_chunks = chunk_graph.predecessors(proxy_chunk)
                assert proxy_chunk.op.n_mappers == len(mapper_chunks), (
                    proxy_chunk.op.n_mappers,
                    mapper_chunks,
                )
                # If some reducer data are not used by downstream, then it won't be included in the chunk graph.
                assert proxy_chunk.op.n_reducers >= n_reducers, (
                    proxy_chunk.op.n_reducers,
                    n_reducers,
                )
        return subtask_graph

    @classmethod
    def _check_shuffle_reduce_chunks(cls, chunks: List, checked_chunks):
        """Check shuffle reduce chunks sorted reducer_index consistent with reducer_ordinal. So shuffle mapper blocks
        can be sorted by reducer_index, and the reducer can fetch mapper data by reducer_ordinal.
        """
        chunks = [c for c in chunks or [] if c not in checked_chunks]
        if not chunks:
            return
        from .....core.operand import MapReduceOperand, ShuffleProxy, OperandStage

        reduce_chunks = defaultdict(list)
        for c in chunks:
            checked_chunks.add(c)
            if isinstance(c.op, MapReduceOperand) and c.op.stage == OperandStage.reduce:
                shuffle_proxies = [
                    c for c in c.inputs if isinstance(c.op, ShuffleProxy)
                ]
                assert len(shuffle_proxies) == 1, (c.inputs, shuffle_proxies)
                reduce_chunks[shuffle_proxies[0]].append(c)
            else:
                cls._check_shuffle_reduce_chunks(c.inputs, checked_chunks)
        for _, reduce_chunks in reduce_chunks.items():
            sorted_chunks_by_indices = sorted(
                reduce_chunks, key=lambda c: c.op.reducer_index
            )
            sorted_chunks_by_ordinals = sorted(
                reduce_chunks, key=lambda c: c.op.reducer_ordinal
            )
            for c1, c2 in zip(sorted_chunks_by_indices, sorted_chunks_by_ordinals):
                assert c1.op.reducer_index == c2.op.reducer_index, (
                    sorted_chunks_by_indices,
                    sorted_chunks_by_ordinals,
                )
            for c in reduce_chunks:
                cls._check_shuffle_reduce_chunks(c.inputs, checked_chunks)
