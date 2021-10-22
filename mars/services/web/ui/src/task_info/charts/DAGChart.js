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
          color: '#FFFFFF',
        },
        {
          text: 'Running',
          color: '#F4B400',
        },
        {
          text: 'Succeeded',
          color: '#00CD95',
        },
        {
          text: 'Failed',
          color: '#E74C3C',
        },
        {
          text: 'Cancelled',
          color: '#BFC9CA',
        },
      ],

      inputNodeColor: '#3281a8',
      outputNodeColor: '#c334eb',
    };
    this.lastSelectedId = null;
  }

  componentDidMount() {
    this.g = new dagGraphLib.Graph().setGraph({});
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  getNodeStyle(nodeId) {
    if (nodeId !== this.lastSelectedId) {
      return 'cursor: pointer; stroke: #333; fill: url(#progress-' + nodeId + ')';
    } else {
      return 'cursor: pointer; stroke: #ff5252; stroke-width: 3; fill: url(#progress-' + nodeId + ')';
    }
  }

  /* eslint no-unused-vars: ["error", { "args": "none" }] */
  componentDidUpdate(prevProps, prevStates, snapshot) {
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
      d3Select('#' + this.props.graphName).selectAll('*').remove();

      // Set up an SVG group so that we can translate the final graph.
      const svg = d3Select('#' + this.props.graphName),
        inner = svg.append('g');

      this.g = new dagGraphLib.Graph().setGraph({});

      // Add the nodes to DAG
      this.props.nodes.forEach((node) => {
        const nodeDetail = this.props.nodesStatus[node.id];
        const value = { node };

        if (this.props.graphName === 'tileableGraph') {
          const nameEndIndex = node.name.indexOf('key') - 1;
          value.label = node.name.substring(0, nameEndIndex);
        } else if (this.props.graphName === 'subtaskGraph') {
          value.label = '';
        }

        if (this.props.graphName === 'tileableGraph') {
          value.rx = value.ry = 5;
        } else if (this.props.graphName === 'subtaskGraph') {
          value.r = 20;
        }

        this.g.setNode(node.id, value);

        /**
         * Add the progress color using SVG linear gradient. The offset on
         * the first stop on the linear gradient marks how much of the node
         * should be filled with color. The second stop adds a white color to
         * the rest of the node
         */
        let nodeProgressGradient = inner.append('linearGradient')
          .attr('id', 'progress-' + node.id);

        const dagNode = this.g.node(node.id);

        /**
         * apply the linear gradient and other css properties
         * to nodes.
         */
        if (nodeDetail === undefined) {
          nodeProgressGradient.append('stop')
            .attr('id', 'progress-' + node.id + '-stop')
            .attr('stop-color', '#FFFFFF')
            .attr('offset', 1);
        } else if (nodeDetail.status === -1 && nodeDetail.progress === -1) {
          nodeProgressGradient.append('stop')
            .attr('id', 'progress-' + node.id + '-stop')
            .attr('stop-color', this.state.inputNodeColor)
            .attr('offset', 1);
        } else if (nodeDetail.status === -2 && nodeDetail.progress === -2) {
          nodeProgressGradient.append('stop')
            .attr('id', 'progress-' + node.id + '-stop')
            .attr('stop-color', this.state.outputNodeColor)
            .attr('offset', 1);
        } else {
          nodeProgressGradient.append('stop')
            .attr('id', 'progress-' + node.id + '-stop')
            .attr('stop-color', this.state.nodeStatusMap[nodeDetail.status].color)
            .attr('offset', nodeDetail.progress);
        }

        nodeProgressGradient.append('stop')
          .attr('stop-color', '#FFFFFF')
          .attr('offset', '0');

        dagNode.shape = this.props.nodeShape;
        dagNode.style = this.getNodeStyle(node.id);
        dagNode.labelStyle = 'cursor: pointer';
      });

      /**
       * Adds edges to the DAG. If an edge has a linkType of 1,
       * the edge will be a dashed line.
       */
      this.props.dependencies.forEach((dependency) => {
        if (dependency.linkType && dependency.linkType === 1) {
          this.g.setEdge(
            dependency.fromNodeId,
            dependency.toNodeId,
            {
              style: 'stroke: #333; fill: none; stroke-dasharray: 5, 5;'
            }
          );
        } else {
          this.g.setEdge(
            dependency.fromNodeId,
            dependency.toNodeId,
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

      if (this.props.nodes.length !== 0) {
        // Run the renderer. This is what draws the final graph.
        render(inner, this.g);
      }

      // onClick function for the tileable
      const handleClick = (e, dagNodeId) => {
        if (this.props.onNodeClick) {
          const selectedNode = this.props.nodes.filter(
            (node) => node.id === dagNodeId
          )[0];
          const nodeDetail = this.props.nodesStatus[selectedNode.id];
          selectedNode['properties'] = nodeDetail['properties'];

          if (dagNodeId !== this.lastSelectedId) {
            const lastSelectedId = this.lastSelectedId;
            this.lastSelectedId = dagNodeId;

            if (lastSelectedId !== null) {
              this.g.node(lastSelectedId).style = this.getNodeStyle(lastSelectedId);
            }
            this.g.node(dagNodeId).style = this.getNodeStyle(dagNodeId);
            render(inner, this.g);
          }

          this.props.onNodeClick(e, selectedNode);
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

      d3Select('#' + this.props.graphName).select('.output').attr(
        'transform',
        'translate(' + (fullWidth - width * initialScale) / 2 + ', ' + (fullHeight - height * initialScale) / 2 + ')'
      );

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
     * If the nodes and dependencies didn't change and
     * only the tileable status changed, we know this is the
     * old graph with updated tileable status, so we just
     * need to update the color of nodes and the progress bar
     */
    if (prevProps.nodes === this.props.nodes
            && prevProps.dependencies === this.props.dependencies
            && prevProps.nodesStatus !== this.props.nodesStatus) {
      const svg = d3Select('#' + this.props.graphName);

      this.props.nodes.forEach((node) => {
        const nodeDetail = this.props.nodesStatus[node.id];

        if (nodeDetail !== undefined && nodeDetail.status >= 0 && nodeDetail.progress >= 0) {
          const dagNode = this.g.node(node.id);

          if (dagNode !== undefined) {
            if (dagNode.style === 'visibility: hidden') {
              dagNode.shape = this.props.nodeShape;
              dagNode.style = 'cursor: pointer; stroke: #333; fill: url(#progress-' + node.id + ')';
              dagNode.labelStyle = 'cursor: pointer';
            }

            svg.select('#progress-' + node.id + '-stop')
              .attr('stop-color', this.state.nodeStatusMap[nodeDetail.status].color)
              .attr('offset', nodeDetail.progress);
          }
        }
      });
    }
  }

  render() {
    return (
      <svg
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
