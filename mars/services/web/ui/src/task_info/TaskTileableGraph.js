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
            tileableStatus: [
                {
                    text: 'Pending',
                    color: '#FFFFFF',
                    legendDotXLoc: '430',
                    legendDotYLoc: '20',
                    legendTextXLoc: '440',
                    legendTextYLoc: '21',
                },
                {
                    text: 'Running',
                    color: '#F4B400',
                    legendDotXLoc: '10',
                    legendDotYLoc: '20',
                    legendTextXLoc: '20',
                    legendTextYLoc: '21',
                },
                {
                    text: 'Succeeded',
                    color: '#00CD95',
                    legendDotXLoc: '105',
                    legendDotYLoc: '20',
                    legendTextXLoc: '115',
                    legendTextYLoc: '21',
                },
                {
                    text: 'Failed',
                    color: '#E74C3C',
                    legendDotXLoc: '345',
                    legendDotYLoc: '20',
                    legendTextXLoc: '355',
                    legendTextYLoc: '21',
                },
                {
                    text: 'Cancelled',
                    color: '#BFC9CA',
                    legendDotXLoc: '225',
                    legendDotYLoc: '20',
                    legendTextXLoc: '235',
                    legendTextYLoc: '21',
                },
            ]
        };
    }

    fetchGraphDetail() {
        const { sessionId, taskId } = this.props;

        fetch(`api/session/${sessionId}/task/${taskId
        }/tileable_graph?action=get_tileable_graph_as_json`)
            .then(res => res.json())
            .then((res) => {
                this.setState({
                    tileables: res.tileables,
                    dependencies: res.dependencies,
                });
            });
    }

    fetchTileableDetail() {
        const { sessionId, taskId } = this.props;

        fetch(`api/session/${sessionId}/task/${taskId
        }/tileable_detail`)
            .then(res => res.json())
            .then((res) => {
                this.setState({
                    tileableDetails: res,
                })
            });
    }

    componentDidMount() {
        this.g = new dagGraphLib.Graph().setGraph({});

        if (this.interval !== undefined) {
            clearInterval(this.interval);
        }
        this.interval = setInterval(() => this.fetchTileableDetail(), 5000);
        this.fetchTileableDetail();
        this.fetchGraphDetail();
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    /**
     * Creates one status entry for the legend of DAG
     *
     * @param {*} svgContainer - The SVG container that the legend will be placed in
     * @param {*} dotX - X coordinate of the colored dot for the legend entry
     * @param {*} dotY - Y coordinate of the colored dot for the legend entry
     * @param {*} textX - X coordinate of the label for the legend entry
     * @param {*} textY - Y coordinate of the label for the legend entry
     * @param {*} color - Status color for the legend entry
     * @param {*} text - Label for the legend entry
     */
    generateGraphLegendItem(svgContainer, dotX, dotY, textX, textY, color, text) {
        if (color === '#FFFFFF') {
            // add an additional stroke so
            // the white color can be visited
            svgContainer
                .append('circle')
                .attr('cx', dotX)
                .attr('cy', dotY)
                .attr('r', 6)
                .attr('stroke', '#333')
                .style('fill', color);
        } else {
            svgContainer
                .append('circle')
                .attr('cx', dotX)
                .attr('cy', dotY)
                .attr('r', 6)
                .style('fill', color);
        }

        svgContainer
            .append('text')
            .attr('x', textX)
            .attr('y', textY)
            .text(text)
            .style('font-size', '15px')
            .attr('alignment-baseline', 'middle');
    }

    /* eslint no-unused-vars: ["error", { "args": "none" }] */
    componentDidUpdate(prevProps, prevStates, snapshot) {
        if (Object.keys(this.state.tileableDetails).length !== this.state.tileables.length) {
            return;
        }

        /**
         * If the tileables and dependencies are different, this is a
         * new DAG, so we will erase everything from the canvas and
         * generate a new dag
         */
        if (prevStates.tileables !== this.state.tileables
            && prevStates.dependencies !== this.state.dependencies) {
            d3Select('#svg-canvas').selectAll('*').remove();

            // Set up an SVG group so that we can translate the final graph.
            const svg = d3Select('#svg-canvas'),
                inner = svg.append('g');

            this.g = new dagGraphLib.Graph().setGraph({});

            // Create the legend for DAG
            const legendSVG = d3Select('#legend');
            this.state.tileableStatus.forEach((status) => this.generateGraphLegendItem(
                legendSVG,
                status.legendDotXLoc,
                status.legendDotYLoc,
                status.legendTextXLoc,
                status.legendTextYLoc,
                status.color,
                status.text
            ));

            // Add the tileables to DAG
            this.state.tileables.forEach((tileable) => {
                const value = { tileable };
                const tileableDetail = this.state.tileableDetails[tileable.tileableId];
                const nameEndIndex = tileable.tileableName.indexOf('key') - 1;

                value.label = tileable.tileableName.substring(0, nameEndIndex);
                value.rx = value.ry = 5;
                this.g.setNode(tileable.tileableId, value);

                /**
                 * Add the progress color using SVG linear gradient. The offset on
                 * the first stop on the linear gradient marks how much of the node
                 * should be filled with color. The second stop adds a white color to
                 * the rest of the node
                 */
                var nodeProgressGradient = inner.append('linearGradient')
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
                node.style = `
                    cursor: pointer;
                    stroke: #333;
                    fill: url(#progress-` + tileable.tileableId + `)`;
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

            // Set up zoom support
            const zoom = d3Zoom().on('zoom', function (e) {
                inner.attr('transform', e.transform);
            });
            svg.call(zoom);

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
            })
        }
    }

    render() {
        return (
            <React.Fragment>
                <svg
                    id='legend'
                    style={{ marginLeft: '6%', width: '90%', height: '10%' }}
                />
                <svg
                    id='svg-canvas'
                    style={{ margin: 30, width: '90%', height: '80%' }}
                />
            </React.Fragment>
        );
    }
}

TaskTileableGraph.propTypes = {
    sessionId: PropTypes.string.isRequired,
    taskId: PropTypes.string.isRequired,
    onTileableClick: PropTypes.func,
};
