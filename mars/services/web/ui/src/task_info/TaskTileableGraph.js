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

        const tileableStatus = ['succeeded', 'running', 'cancelled', 'failed', 'pending'];

        fetch(`api/session/${sessionId}/task/${taskId
        }/tileable_graph?action=get_tileable_graph_as_json`)
            .then(res => res.json())
            .then((res) => {
                console.log(res);
                let newTileableDetails = {};
                for (let i = 0; i < res.tileables.length; i++) {
                    newTileableDetails[res.tileables[i].tileable_id] = {
                        progress: Math.random(),
                        status: tileableStatus[Math.floor(Math.random()*tileableStatus.length)],
                    };
                }

                this.setState({
                    tileables: res.tileables,
                    dependencies: res.dependencies,
                    tileableDetails: newTileableDetails,
                });
            });
    }

    componentDidMount() {
        this.g = new dagGraphLib.Graph().setGraph({});
        this.fetchGraphDetail();
    }

    /* eslint no-unused-vars: ["error", { "args": "none" }] */
    componentDidUpdate(prevProps, prevStates, snapshot) {
        if (prevStates.tileables !== this.state.tileables
                && prevStates.dependencies !== this.state.dependencies) {
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

                let nameEndIndex = tileable.tileable_name.indexOf('key') - 1;
                value.label = tileable.tileable_name.substring(0, nameEndIndex);
                value.rx = value.ry = 5;
                this.g.setNode(tileable.tileable_id, value);

                const tileableDetail = this.state.tileableDetails[tileable.tileable_id];

                var nodeProgressGradient = inner.append('linearGradient')
                    .attr('id', 'progress-' + tileable.tileable_id);

                if (tileableDetail.status === 'pending') {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#ffffff')
                    .attr('offset', '0%');
                } else if (tileableDetail.status === 'running') {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#f4b400')
                    .attr('offset', tileableDetail.progress * 100 + '%');
                } else if (tileableDetail.status === 'succeeded') {
                    nodeProgressGradient.append('stop')
                    .attr('stop-color', '#00ff00')
                    .attr('offset', '100%');
                } else if (tileableDetail.status === 'failed') {
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
                const node = this.g.node(tileable.tileable_id);
                node.style = `
                    cursor: pointer;
                    stroke: #333;
                    fill: url(#progress-` + tileable.tileable_id + `)`;
                node.labelStyle = 'cursor: pointer';
            });

            this.state.dependencies.forEach((dependency) => {
                // In future label may be named based on linkType?
                this.g.setEdge(
                    dependency.from_tileable_id,
                    dependency.to_tileable_id,
                    { label: '' }
                );
            });

            let gInstance = this.g;
            // Round the corners of the nodes
            gInstance.nodes().forEach(function (v) {
                const node = gInstance.node(v);
                node.rx = node.ry = 5;
            });

            //makes the lines smooth
            gInstance.edges().forEach(function (e) {
                const edge = gInstance.edge(e.v, e.w);
                edge.style = 'stroke: #333; fill: none';
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
                        (tileable) => tileable.tileable_id == node
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
