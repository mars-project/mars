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


export default class SubtaskGraph extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            subtasks: [],
            dependencies: [],
        };
    }

    fetchGraphDetail() {
        const { sessionId, taskId, tileableId } = this.props;

        if (sessionId === undefined || taskId === undefined || tileableId === undefined) {
            return;
        }

        fetch(`api/session/${sessionId}/task/${taskId
        }/${tileableId}/subtasks`)
            .then(res => res.json())
            .then((res) => {
                this.setState({
                    subtasks: res.subtasks,
                    dependencies: res.dependencies,
                });
            });
    }

    componentDidMount() {
        this.g = new dagGraphLib.Graph().setGraph({});

        if (this.interval !== undefined) {
            clearInterval(this.interval);
        }
        this.interval = setInterval(() => this.fetchGraphDetail(), 5000);
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    /* eslint no-unused-vars: ["error", { "args": "none" }] */
    componentDidUpdate(prevProps, prevStates, snapshot) {
        if (prevStates.subtasks !== this.state.subtasks
            && prevStates.dependencies !== this.state.dependencies) {
            d3Select('#svg-canvas').selectAll('*').remove();

            // Set up an SVG group so that we can translate the final graph.
            const svg = d3Select('#svg-canvas'),
                inner = svg.append('g');

            this.g = new dagGraphLib.Graph().setGraph({});

            // Add the subtasks to DAG
            this.state.subtasks.forEach((subtask) => {
                const value = { subtask };
                value.rx = value.ry = 5;
                this.g.setNode(subtask.subtaskId, value);

                /**
                 * Add the progress color using SVG linear gradient. The offset on
                 * the first stop on the linear gradient marks how much of the node
                 * should be filled with color. The second stop adds a white color to
                 * the rest of the node
                 */
                let nodeProgressGradient = inner.append('linearGradient')
                    .attr('id', 'progress-' + tileable.tileableId);

                nodeProgressGradient.append('stop')
                    .attr('id', 'progress-' + tileable.tileableId + '-stop')
                    .attr('stop-color', this.state.tileableStatus[tileableDetail.status].color)
                    .attr('offset', tileableDetail.progress);

                nodeProgressGradient.append('stop')
                    .attr('stop-color', '#FFFFFF')
                    .attr('offset', '0');

                /**
                 * apply the linear gradient and other css properties
                 * to nodes.
                 */
                const node = this.g.node(tileable.tileableId);
                node.style = 'cursor: pointer; stroke: #333; fill: url(#progress-' + tileable.tileableId + ')';
                node.labelStyle = 'cursor: pointer';
            });

            /**
             * Adds edges to the DAG. If an edge has a linkType of 1,
             * the edge will be a dashed line.
             */
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

            if (this.state.tileables.length !== 0) {
                // Run the renderer. This is what draws the final graph.
                render(inner, this.g);
            }

            // onClick function for the tileable
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
            const bounds = inner.node().getBBox();
            const parent = inner.node().parentElement;
            const width = bounds.width,
                height = bounds.height;
            const fullWidth = parent.clientWidth,
                fullHeight = parent.clientHeight;
            const initialScale = fullHeight >= height ? 1 : fullHeight / height;

            d3Select('.output').attr('transform', 'translate(' + (fullWidth - width * initialScale) / 2 + ', ' + (fullHeight - height * initialScale) / 2 + ')');

            // Set up zoom support
            const zoom = d3Zoom().on('zoom', function (e) {
                inner.attr('transform', e.transform);
            });

            svg.call(
                zoom,
                zoom.transform,
                d3ZoomIdentity.scale(initialScale)
            );

            if (this.g.graph() !== null || this.g.graph() !== undefined) {
                svg.attr('height', this.g.graph().height * initialScale + 40);
            } else {
                svg.attr('height', 40);
            }
        }

        /**
         * If the tileables and dependencies didn't change and
         * only the tileable status changed, we know this is the
         * old graph with updated tileable status, so we just
         * need to update the color of nodes and the progress bar
         */
        if (prevStates.tileables === this.state.tileables
            && prevStates.dependencies === this.state.dependencies
            && prevStates.tileableDetails !== this.state.tileableDetails) {
            const svg = d3Select('#svg-canvas');
            this.state.tileables.forEach((tileable) => {
                const tileableDetail = this.state.tileableDetails[tileable.tileableId];

                svg.select('#progress-' + tileable.tileableId + '-stop')
                    .attr('stop-color', this.state.tileableStatus[tileableDetail.status].color)
                    .attr('offset', tileableDetail.progress);
            });
        }
    }

    render() {
        return (
            <React.Fragment>
                <svg
                    id='svg-canvas'
                    style={{ margin: 30, width: '90%', height: '80%' }}
                />
            </React.Fragment>
        );
    }
}

SubtaskGraph.propTypes = {
    sessionId: PropTypes.string.isRequired,
    taskId: PropTypes.string.isRequired,
    tileableId: PropTypes.string.isRequired,
};
