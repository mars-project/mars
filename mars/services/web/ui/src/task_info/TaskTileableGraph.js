/*
 * Copyright 1999-2021 Alibaba Group Holding Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React from 'react';
import { select as d3Select } from 'd3-selection';
import {
    zoom as d3Zoom,
    zoomIdentity as d3ZoomIdentity
} from 'd3-zoom';
import {
    graphlib as dagGraphLib,
    render as DagRender
} from 'dagre-d3';
import PropTypes from 'prop-types';


export default class TaskTileableGraph extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedTileable: null,
            tileables: [],
            dependencies: [],
            tileableDetails: {},
        };
    }

    fetchGraphDetail() {
        const { sessionId, taskId } = this.props;

        fetch(`api/session/${sessionId}/task/${taskId
        }/tileable_detail`)
            .then(res => res.json())
            .then((res) => {
                console.log('res', res, Object.keys(res).length)
                this.setState({
                    tileableDetails: res,
                })
            });

        fetch(`api/session/${sessionId}/task/${taskId
        }/tileable_graph?action=get_tileable_graph_as_json`)
            .then(res => res.json())
            .then((res) => {
                console.log('res', res, res.tileables.length);
                this.setState({
                    tileables: res.tileables,
                    dependencies: res.dependencies,
                });
            });
    }

    componentDidMount() {
        this.g = new dagGraphLib.Graph().setGraph({});
        this.fetchGraphDetail();
    }

    /* eslint no-unused-vars: ["error", { "args": "none" }] */
    componentDidUpdate(prevProps, prevStates, snapshot) {
        if (Object.keys(this.state.tileableDetails).length !== this.state.tileables.length) {
            return;
        }

        if ((prevStates.tileables !== this.state.tileables
                && prevStates.dependencies !== this.state.dependencies)
                || prevStates.tileableDetails !== this.state.tileableDetails) {
            d3Select('#svg-canvas').selectAll('*').remove();

            // Set up an SVG group so that we can translate the final graph.
            const svg = d3Select('#svg-canvas'),
                inner = svg.append('g');

            svg
                .append("circle")
                .attr("cx", 400)
                .attr("cy", 20)
                .attr("r", 6)
                .style("fill", "#f4b400");
            svg
                .append("circle")
                .attr("cx", 400)
                .attr("cy", 40)
                .attr("r", 6)
                .style("fill", "#00ff00");
            svg
                .append("circle")
                .attr("cx", 400)
                .attr("cy", 60)
                .attr("r", 6)
                .style("fill", "#808080");
            svg
                .append("circle")
                .attr("cx", 400)
                .attr("cy", 80)
                .attr("r", 6)
                .style("fill", "#ff0000");
            svg
                .append("circle")
                .attr("cx", 400)
                .attr("cy", 100)
                .attr("r", 6)
                .attr("stroke", "#333")
                .style("fill", "#ffffff");

            svg
                .append("text")
                .attr("x", 420)
                .attr("y", 100)
                .text("Pending")
                .style("font-size", "15px")
                .attr("alignment-baseline", "middle");
            svg
                .append("text")
                .attr("x", 420)
                .attr("y", 20)
                .text("Running")
                .style("font-size", "15px")
                .attr("alignment-baseline", "middle");
            svg
                .append("text")
                .attr("x", 420)
                .attr("y", 40)
                .text("Succeeded")
                .style("font-size", "15px")
                .attr("alignment-baseline", "middle");
            svg
                .append("text")
                .attr("x", 420)
                .attr("y", 60)
                .text("Cancelled")
                .style("font-size", "15px")
                .attr("alignment-baseline", "middle");
            svg
                .append("text")
                .attr("x", 420)
                .attr("y", 80)
                .text("Failed")
                .style("font-size", "15px")
                .attr("alignment-baseline", "middle");


            this.g = new dagGraphLib.Graph().setGraph({});
            this.state.tileables.forEach((tileable) => {
                const value = { tileable };

                let nameEndIndex = tileable.tileableName.indexOf('key') - 1;
                value.label = tileable.tileableName.substring(0, nameEndIndex);
                value.rx = value.ry = 5;
                this.g.setNode(tileable.tileableId, value);

                const tileableDetail = this.state.tileableDetails[tileable.tileableId];

                var nodeProgressGradient = inner.append('linearGradient')
                    .attr('id', 'progress-' + tileable.tileableId);

                if (tileableDetail.status === 0) {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#ffffff')
                    .attr('offset', '0%');
                } else if (tileableDetail.status === 1) {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#f4b400')
                    .attr('offset', tileableDetail.progress * 100 + '%');
                } else if (tileableDetail.status === 2) {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#00ff00')
                    .attr('offset', '100%');
                } else if (tileableDetail.status === 3) {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#ff0000')
                    .attr('offset', tileableDetail.progress * 100 + '%');
                } else {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#808080')
                    .attr('offset', tileableDetail.progress * 100 + '%');
                }

                nodeProgressGradient.append('stop')
                    .attr('stop-color', '#ffffff')
                    .attr('offset', '0%');

                // In future fill color based on progress
                const node = this.g.node(tileable.tileableId);
                node.style = `
                    cursor: pointer;
                    stroke: #333;
                    fill: url(#progress-` + tileable.tileableId + `)`;
                 node.labelStyle = 'cursor: pointer';
            });

            this.state.dependencies.forEach((dependency) => {
                if (dependency.linkType === 1) {
                    this.g.setEdge(
                        dependency.fromTileableId,
                        dependency.toTileableId,
                        {
                            style: 'stroke: #333; fill: none; stroke-dasharray: 5, 5;'
                        }
                    );
                } else {
                    this.g.setEdge(
                        dependency.fromTileableId,
                        dependency.toTileableId,
                        {
                            style: 'stroke: #333; fill: none;'
                        }
                    );
                }

            });

            let gInstance = this.g;
            // Round the corners of the nodes
            gInstance.nodes().forEach(function (v) {
                const node = gInstance.node(v);
                node.rx = node.ry = 5;
            });

            // Create the renderer
            const render = new DagRender();

            // Set up zoom support
            const zoom = d3Zoom().on('zoom', function (e) {
                inner.attr('transform', e.transform);
            });
            svg.call(zoom);

            if (this.state.tileables.length !== 0) {
                // Run the renderer. This is what draws the final graph.
                render(inner, this.g);
            }

            const handleClick = (e, node) => {
                if (this.props.onTileableClick) {
                    const selectedTileable = this.state.tileables.filter(
                        (tileable) => tileable.tileableId == node
                    )[0];
                    this.props.onTileableClick(e, selectedTileable);
                }
            };

            inner.selectAll('g.node').on('click', handleClick);

            // Center the graph
            const initialScale = 0.9;
            svg.call(
                zoom.transform,
                d3ZoomIdentity.scale(initialScale)
            );

            if (this.g.graph() !== null || this.g.graph() !== undefined) {
                svg.attr('height', this.g.graph().height * initialScale + 40);
            } else {
                svg.attr('height', 40);
            }
        }
    }

    render() {
        return (
            <svg
                id="svg-canvas"
                style={{ margin: 30, width: '90%' }}
            />
        );
    }
}

TaskTileableGraph.propTypes = {
    sessionId: PropTypes.string.isRequired,
    taskId: PropTypes.string.isRequired,
    onTileableClick: PropTypes.func,
};
