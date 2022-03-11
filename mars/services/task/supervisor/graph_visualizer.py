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
from ...subtask import Subtask
from .processor import TaskProcessor


class GraphVisualizer:
    task_processor: TaskProcessor

    def __init__(self, task_processor):
        self.task_processor = task_processor

    def to_dot(self):
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
        for stage_line in itertools.combinations(
            range(len(self.task_processor.stage_processors))[::-1], 2
        ):
            line_colors[stage_line] = f'"/spectral9/{next(color_iter)}"'

        for stage_processor in self.task_processor.stage_processors:
            for subtask in stage_processor.subtask_graph.topological_iter():
                current_cluster = f"cluster_{subgraph_index}"
                sio.write(
                    self._export_subtask_to_dot(
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
                ):
                    node_name = f'"Chunk:{input_chunk.key[:trunc_key]}"'
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
                        f'"Chunk:{input_chunk.key[:trunc_key]}" -> "{op_name}:{op.key[:trunc_key]}" '
                        f"[lhead={subgraph_name} ltail={tail_cluster} {line_style}];\n"
                    )
                else:
                    sio.write(
                        f'"Chunk:{input_chunk.key[:trunc_key]}" -> "{op_name}:{op.key[:trunc_key]}"\n'
                    )

            for output_chunk in op.outputs or []:
                if output_chunk.key not in visited:
                    node_name = f'"Chunk:{output_chunk.key[:trunc_key]}"'
                    sio.write(f"{node_name} {chunk_style}\n")
                    all_nodes.append(node_name)
                    visited.add(output_chunk.key)
                if op.key not in visited:
                    node_name = f'"{op_name}:{op.key[:trunc_key]}"'
                    sio.write(f"{node_name} {operand_style}\n")
                    all_nodes.append(node_name)
                    visited.add(op.key)
                sio.write(
                    f'"{op_name}:{op.key[:trunc_key]}" -> "Chunk:{output_chunk.key[:5]}"\n'
                )
        # write subgraph info
        sio.write(f"subgraph {subgraph_name} {{\n")
        nodes_str = " ".join(all_nodes)
        sio.write(f"{nodes_str};\n")
        sio.write(f'label="{subtask.subtask_id}";\n}}')
        sio.write("\n")
        return sio.getvalue()
