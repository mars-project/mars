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
from io import StringIO
from typing import Dict, List

from ....core.operand import Fetch, FetchShuffle
from ...subtask import Subtask, SubtaskGraph


class GraphVisualizer:
    @classmethod
    def to_dot(cls, subtask_graphs: List[SubtaskGraph]):
        sio = StringIO()
        sio.write("digraph {\n")
        sio.write("splines=curved\n")
        sio.write("rankdir=BT\n")
        sio.write("graph [compound=true];\n")
        subgraph_index = 0
        current_stage = 0
        result_chunk_to_subtask = dict()
        line_colors = dict()
        color_iter = iter(itertools.cycle(range(1, 9)))
        for stage_line in itertools.combinations(range(len(subtask_graphs))[::-1], 2):
            line_colors[stage_line] = f'"/spectral9/{next(color_iter)}"'

        for subtask_graph in subtask_graphs:
            for subtask in subtask_graph.topological_iter():
                current_cluster = f"cluster_{subgraph_index}"
                sio.write(
                    cls._export_subtask_to_dot(
                        subtask,
                        current_cluster,
                        current_stage,
                        line_colors,
                        result_chunk_to_subtask,
                    )
                )
                for c in subtask.chunk_graph.results:
                    result_chunk_to_subtask[c.key] = [current_stage, current_cluster]
                subgraph_index += 1
            current_stage += 1
        sio.write("}")
        return sio.getvalue()

    @classmethod
    def _gen_chunk_key(cls, chunk, trunc_key):
        if "_" in chunk.key:
            key, index = chunk.key.split("_", 1)
            return "_".join([key[:trunc_key], index])
        else:  # pragma: no cover
            return chunk.key[:trunc_key]

    @classmethod
    def _export_subtask_to_dot(
        cls,
        subtask: Subtask,
        subgraph_name: str,
        current_stage: int,
        line_colors: Dict,
        chunk_key_to_subtask: Dict[str, List],
        trunc_key: int = 5,
    ):

        chunk_graph = subtask.chunk_graph
        sio = StringIO()
        chunk_style = "[shape=box]"
        operand_style = "[shape=circle]"

        visited = set()
        all_nodes = []
        for node in chunk_graph.iter_nodes():
            op = node.op
            if isinstance(node.op, (Fetch, FetchShuffle)):
                continue
            op_name = type(op).__name__
            if op.stage is not None:
                op_name = f"{op_name}:{op.stage.name}"
            if op.key in visited:
                continue
            for input_chunk in op.inputs or []:
                if input_chunk.key not in visited and not isinstance(
                    input_chunk.op, (Fetch, FetchShuffle)
                ):  # pragma: no cover
                    node_name = f'"Chunk:{cls._gen_chunk_key(input_chunk, trunc_key)}"'
                    sio.write(f"{node_name} {chunk_style}\n")
                    all_nodes.append(node_name)
                    visited.add(input_chunk.key)
                if op.key not in visited:
                    node_name = f'"{op_name}:{op.key[:trunc_key]}"'
                    sio.write(f"{node_name} {operand_style}\n")
                    all_nodes.append(node_name)
                    visited.add(op.key)
                if (
                    isinstance(input_chunk.op, (Fetch, FetchShuffle))
                    and input_chunk.key in chunk_key_to_subtask
                ):
                    stage, tail_cluster = chunk_key_to_subtask[input_chunk.key]
                    if stage == current_stage:
                        line_style = "style=bold"
                    else:
                        line_style = (
                            f"style=bold color={line_colors[(current_stage, stage)]}"
                        )
                    sio.write(
                        f'"Chunk:{cls._gen_chunk_key(input_chunk, trunc_key)}" ->'
                        f' "{op_name}:{op.key[:trunc_key]}" '
                        f"[lhead={subgraph_name} ltail={tail_cluster} {line_style}];\n"
                    )
                else:
                    sio.write(
                        f'"Chunk:{cls._gen_chunk_key(input_chunk, trunc_key)}" -> '
                        f'"{op_name}:{op.key[:trunc_key]}"\n'
                    )

            for output_chunk in op.outputs or []:
                if output_chunk.key not in visited:
                    node_name = f'"Chunk:{cls._gen_chunk_key(output_chunk, trunc_key)}"'
                    sio.write(f"{node_name} {chunk_style}\n")
                    all_nodes.append(node_name)
                    visited.add(output_chunk.key)
                if op.key not in visited:
                    node_name = f'"{op_name}:{op.key[:trunc_key]}"'
                    sio.write(f"{node_name} {operand_style}\n")
                    all_nodes.append(node_name)
                    visited.add(op.key)
                sio.write(
                    f'"{op_name}:{op.key[:trunc_key]}" -> '
                    f'"Chunk:{cls._gen_chunk_key(output_chunk, trunc_key)}"\n'
                )
        # write subgraph info
        sio.write(f"subgraph {subgraph_name} {{\n")
        nodes_str = " ".join(all_nodes)
        sio.write(f"{nodes_str};\n")
        sio.write(f'label="{subtask.subtask_id}";\n}}')
        sio.write("\n")
        return sio.getvalue()
