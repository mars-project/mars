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
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import PropTypes from 'prop-types';
import {
    nodeType,
    nodesStatusType,
    dependencyType,
    dagType,
} from './DAGPropTypes';


export default class DAGChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            nodeStatusMap: [
                {
                    text: 'Pending',
                    color: 'rgb(255, 255, 255)',
                },
                {
                    text: 'Running',
                    color: 'rgb(240, 180, 0)',
                },
                {
                    text: 'Succeeded',
                    color: 'rgb(0, 205, 149)',
                },
                {
                    text: 'Failed',
                    color: 'rgb(231,76,60)',
                },
                {
                    text: 'Cancelled',
                    color: 'rgb(191,201,202)',
                },
            ],

            inputNodeColor: 'rgb(50,129,168)',
            outputNodeColor: 'rgb(195,52,235)',
        };
    }

    getProgressSVG(progress, status) {
        if (progress === -1 && status === -1) {
            const svg = `
            <svg width="60" height="60" viewBox="0 0 60 60" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="60" height="60" style="fill:${this.state.inputNodeColor};stroke-width:0.3;stroke:rgb(0,0,0)" />
                <rect x="60" y="0" width="0" height="60" style="fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)" />
            </svg>`;

            return encodeURI("data:image/svg+xml;utf-8,"+svg);
        }

        if (progress === -2 && status === -2) {
            const svg = `
            <svg width="60" height="60" viewBox="0 0 60 60" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="60" height="60" style="fill:${this.state.outputNodeColor};stroke-width:0.3;stroke:rgb(0,0,0)" />
                <rect x="60" y="0" width="0" height="60" style="fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)" />
            </svg>`;

            return encodeURI("data:image/svg+xml;utf-8,"+svg);
        }

        const mark = 60 * progress;
        const remain = 60 - mark;

        const svg = `
            <svg width="60" height="60" viewBox="0 0 60 60" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="60" height="60" style="fill:${this.state.nodeStatusMap[status].color};stroke-width:0.3;stroke:rgb(0,0,0)" />
                <rect x="${mark}" y="0" width="${remain}" height="60" style="fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)" />
            </svg>`;

        return encodeURI("data:image/svg+xml;utf-8,"+svg);
    }

    /* eslint no-unused-vars: ["error", { "args": "none" }] */
    componentDidUpdate(prevProps, prevStates, snapshot) {
        cytoscape.use(dagre);

        if (this.props === undefined || this.props.nodes === undefined || this.props.nodes.length === 0) {
            return;
        }

        if (Object.keys(this.props.nodesStatus).length !== this.props.nodes.length) {
            return;
        }

        /**
         * If the nodes and dependencies are different, this is a
         * new DAG, so we will erase everything from the canvas and
         * generate a new dag
         */
        if (prevProps.nodes !== this.props.nodes
            && prevProps.dependencies !== this.props.dependencies) {
                console.log(this.props);

                let dagNodes = [], dagEdges = [], dagStyles = [];

                dagStyles.push({
                    selector: '.linkType-1',
                    style: {
                        'width': 1,
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'line-style': 'dashed',
                    }
                });

                dagStyles.push({
                    selector: '.linkType-0',
                    style: {
                        'width': 1,
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                    }
                });

                this.props.nodes.forEach((node) => {
                    const nodeDetail = this.props.nodesStatus[node.id];

                    const nodeElement = {
                        data: {
                            id: node.id
                        },
                        group: 'nodes',
                    };

                    const nodeStyle = {
                        selector: '#' + node.id,
                        css: {
                            'background-image': this.getProgressSVG(nodeDetail.progress, nodeDetail.status),
                        }
                    };

                    dagNodes.push(nodeElement);
                    dagStyles.push(nodeStyle);
                });

                this.props.dependencies.forEach((dependency) => {
                    const dependencyClass = dependency.linkType === 1 ? 'linkType-1' : 'linkType-0';

                    const dependencyElement = {
                        data: {
                            source: dependency.fromNodeId,
                            target: dependency.toNodeId,
                        },
                        group: 'edges',
                        classes: dependencyClass
                    }
                    dagEdges.push(dependencyElement);
                });

                let cy = window.cy = cytoscape({
                    container: document.getElementById(this.props.graphName),

                    boxSelectionEnabled: false,
                    autounselectify: true,

                    layout: {
                        name: 'dagre'
                    },

                    style: dagStyles,

                    elements: {
                        nodes: dagNodes,
                        edges: dagEdges,
                    }
                });
        }

        /**
         * If the nodes and dependencies didn't change and
         * only the tileable status changed, we know this is the
         * old graph with updated tileable status, so we just
         * need to update the color of nodes and the progress bar
         */
        if (prevProps.nodes === this.props.nodes
            && prevProps.dependencies === this.props.dependencies
            && prevProps.nodesStatus !== this.props.nodesStatus) {
        }
    }

    render() {
        return (
            <div
                id={this.props.graphName}
                style={this.props.dagStyle}
            />
        );
    }
}

DAGChart.propTypes = {
    graphName: PropTypes.string.isRequired,
    dagStyle: dagType,
    nodes: nodeType,
    nodeShape: PropTypes.string.isRequired,
    nodesStatus: nodesStatusType,
    dependencies: dependencyType,
    onNodeClick: PropTypes.func,
};
