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
import { select as d3Select } from 'd3-selection';
import dagre from 'cytoscape-dagre';
import PropTypes from 'prop-types';
import {
  nodeType,
  nodesStatusType,
  dependencyType,
  dagType,
} from './DAGPropTypes';

export default class DAGCanvasChart extends React.Component {
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

      subtaskWidth: 60,
      subtaskHeight: 60,
    };
  }

  getProgressSVG(nodeType, progress, status, width, height) {
    if (nodeType === 'Input') {
      const svg = `
        <svg width='${width}' height='${height}' viewBox='0 0 ${width} ${height}' version='1.1' xmlns='http://www.w3.org/2000/svg'>
          <rect x='0' y='0' width='${width}' height='${height}' style='fill:${this.state.inputNodeColor};stroke-width:0.3;stroke:rgb(0,0,0)' />
          <rect x='${width}' y='0' width='0' height='${height}' style='fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)' />
        </svg>`;

      return encodeURI('data:image/svg+xml;utf-8,'+svg);
    }

    if (nodeType === 'Output') {
      const svg = `
        <svg width='${width}' height='${height}' viewBox='0 0 ${width} ${height}' version='1.1' xmlns='http://www.w3.org/2000/svg'>
          <rect x='0' y='0' width='${width}' height='${height}' style='fill:${this.state.outputNodeColor};stroke-width:0.3;stroke:rgb(0,0,0)' />
          <rect x='${width}' y='0' width='0' height='${height}' style='fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)' />
        </svg>`;

      return encodeURI('data:image/svg+xml;utf-8,'+svg);
    }

    const mark = width * progress;
    const remain = width - mark;

    const svg = `
      <svg width='${width}' height='${height}' viewBox='0 0 ${width} ${height}' version='1.1' xmlns='http://www.w3.org/2000/svg'>
        <rect x='0' y='0' width='${width}' height='${height}' style='fill:${this.state.nodeStatusMap[status].color};stroke-width:0.3;stroke:rgb(0,0,0)' />
        <rect x='${mark}' y='0' width='${remain}' height='${height}' style='fill:rgb(255, 255, 255);stroke-width:0.1;stroke:rgb(0, 0, 0)' />
      </svg>`;

    return encodeURI('data:image/svg+xml;utf-8,'+svg);
  }

  componentDidMount() {
    this.tooltip = d3Select('#node-tooltip')
      .style('opacity', 0)
      .style('background', 'lightsteelblue')
      .style('position', 'absolute')
      .style('width', 'auto')
      .style('height', '100px')
      .style('overflow', 'auto')
      .style('padding', '5px')
      .style('border', '0px')
      .style('border-radius', '8px');
  }

  /* eslint no-unused-vars: ["error", { "args": "none" }] */
  componentDidUpdate(prevProps, prevStates, snapshot) {
    cytoscape.use(dagre);

    if (this.props === undefined || this.props.nodes === undefined || this.props.nodes.length === 0) {
      return;
    }

    /**
     * If the nodes and dependencies are different, this is a
     * new DAG, so we will erase everything from the canvas and
     * generate a new dag
     */
    if (prevProps.nodes !== this.props.nodes
            && prevProps.dependencies !== this.props.dependencies) {
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

        if (nodeDetail === undefined) {
          const nodeElement = {
            data: {
              id: node.id,
            },
            group: 'nodes',
            grabbable: false,
          };

          const nodeStyle = {
            selector: '#' + node.id,
            css: {
              'background-image': this.getProgressSVG(
                'Calculation',
                1,
                0,
                this.state.subtaskWidth,
                this.state.subtaskHeight),
              'border-width': 0.3,
              'border-color': 'black',
            }
          };

          dagNodes.push(nodeElement);
          dagStyles.push(nodeStyle);
        } else {
          const nodeElement = {
            data: {
              id: node.id,
            },
            group: 'nodes',
            grabbable: false,
          };

          const nodeStyle = {
            selector: '#' + node.id,
            css: {
              'background-image': this.getProgressSVG(
                nodeDetail.nodeType,
                nodeDetail.progress,
                nodeDetail.status,
                this.state.subtaskWidth,
                this.state.subtaskHeight),
              'border-width': 0.3,
              'border-color': 'black',
            }
          };

          dagNodes.push(nodeElement);
          dagStyles.push(nodeStyle);
        }
      });

      this.props.dependencies.forEach((dependency) => {
        const dependencyClass = dependency.linkType === 1 ? 'linkType-1' : 'linkType-0';

        const sourceNodeDetail = this.props.nodesStatus[dependency.fromNodeId];
        const targetNodeDetail = this.props.nodesStatus[dependency.toNodeId];

        const dependencyElement = {
          data: {
            source: dependency.fromNodeId,
            target: dependency.toNodeId,
            sourceNodeName: sourceNodeDetail === undefined ? '' : sourceNodeDetail.name,
            targetNodeName: targetNodeDetail === undefined ? '' : targetNodeDetail.name,
          },
          group: 'edges',
          classes: dependencyClass
        };
        dagEdges.push(dependencyElement);
      });

      this.cy = window.cy = cytoscape({
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

      this.cy.on('mouseover', 'node', (e) => {
        const x = e.originalEvent.clientX;
        const y = e.originalEvent.clientY;

        const selectedNodeId = e.target[0]._private.data.id;
        const paths = e.target[0]._private.edges.map((edge) => {
          const dependency = edge._private.data;
          if (dependency.fromNodeId === selectedNodeId) {
            return dependency.targetNodeName;
          } else {
            return dependency.sourceNodeName;
          }
        });

        const nodeDetail = this.props.nodesStatus[selectedNodeId];

        if (nodeDetail !== undefined) {
          if (nodeDetail.nodeType !== 'Calculation') {
            let connectedNodes = '';
            const title = nodeDetail.nodeType === 'Output' ? 'Source Nodes: ' : 'Target Nodes: ';

            for (let i = 0; i < paths.length; i++) {
              connectedNodes += '<p>' + title + paths[i] + '</p>';
            }

            let tooltipInfo = `
              <div>
                <p>Connected Nodes:</p>
                ${connectedNodes}
              </div>
              `;

            this.tooltip.html(tooltipInfo)
              .style('left', (x) + 'px')
              .style('top', (y+5) + 'px');
          } else {
            let tooltipInfo = `
              <div>
                <p>Node Name:</p>
                ${this.props.nodesStatus[selectedNodeId].name}
              </div>
              `;

            this.tooltip.html(tooltipInfo)
              .style('left', (x) + 'px')
              .style('top', (y+5) + 'px');
          }

          this.tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        }
      });

      this.cy.on('mouseout', 'node', (e) => {
        this.tooltip.transition()
          .duration(200)
          .style('opacity', 0);
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
      this.props.nodes.forEach((node) => {
        const nodeDetail = this.props.nodesStatus[node.id];

        if (nodeDetail !== undefined && nodeDetail.status >= 0 && nodeDetail.progress >= 0) {
          this.cy.nodes(`[id = "${node.id}"]`)
            .style('background-image', this.getProgressSVG(
              nodeDetail.nodeType,
              nodeDetail.progress,
              nodeDetail.status,
              this.state.subtaskWidth,
              this.state.subtaskHeight))
            .style('border-width', 0.3)
            .style('border-color', 'black');
        }
      });
    }
  }

  render() {
    return (
      <React.Fragment>
        <div
          id={this.props.graphName}
          style={this.props.dagStyle}
        />
        <div id='node-tooltip' />
      </React.Fragment>
    );
  }
}

DAGCanvasChart.propTypes = {
  graphName: PropTypes.string.isRequired,
  dagStyle: dagType,
  nodes: nodeType,
  nodeShape: PropTypes.string.isRequired,
  nodesStatus: nodesStatusType,
  dependencies: dependencyType,
  onNodeClick: PropTypes.func,
};
