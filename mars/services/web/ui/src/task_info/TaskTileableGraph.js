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
  graphlib as dagGraphLib,
} from 'dagre-d3';
import PropTypes from 'prop-types';
import DAGChart from './charts/DAGChart';


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

  fetchGraphDetail() {
    const { sessionId, taskId } = this.props;

    fetch(`api/session/${sessionId}/task/${taskId
    }/tileable_graph?action=get_tileable_graph_as_json`)
      .then(res => res.json())
      .then((res) => {
        this.setState({
          tileables: res.tileables.map(({tileableId, tileableName}) => {
            return (
              {
                id: tileableId,
                name: tileableName,
              }
            );
          }),
          dependencies: res.dependencies.map(({fromTileableId, toTileableId}) => {
            return (
              {
                fromNodeId: fromTileableId,
                toNodeId: toTileableId,
              }
            );
          }),
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
        });
      });
  }

  componentDidMount() {
    this.g = new dagGraphLib.Graph().setGraph({});

    if (this.interval !== undefined) {
      clearInterval(this.interval);
    }
    this.interval = setInterval(() => this.fetchTileableDetail(), 1000);
    this.fetchTileableDetail();
    this.fetchGraphDetail();

    // Create the legend for DAG
    const legendSVG = d3Select('#tileables-legend');
    this.state.tileableStatus.forEach((status) => this.generateGraphLegendItem(
      legendSVG,
      status.legendDotXLoc,
      status.legendDotYLoc,
      status.legendTextXLoc,
      status.legendTextYLoc,
      status.color,
      status.text
    ));
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  render() {
    const dagStyle = {
      margin: 30,
      width: '90%',
      height: '80%',
    };

    if (this.state === undefined ||
            this.state.tileables === undefined ||
            this.state.dependencies === undefined ||
            this.state.tileableDetails === undefined) {
      return (
        <div>Loading</div>
      );
    }

    return (
      <React.Fragment>
        <svg
          id='tileables-legend'
          style={{ marginLeft: '6%', width: '90%', height: '10%' }}
        />
        <DAGChart
          graphName='tileableGraph'
          dagStyle={dagStyle}
          nodes={this.state.tileables}
          nodeShape='rect'
          nodesStatus={this.state.tileableDetails}
          dependencies={this.state.dependencies}
          onNodeClick={this.props.onTileableClick}
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
