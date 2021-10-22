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
import PropTypes from 'prop-types';
import { select as d3Select } from 'd3-selection';
import DAGCanvasChart from './charts/DAGCanvasChart';

export default class SubtaskGraph extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      subtasks: [],
      dependencies: [],
      subtaskDetails: {},
      subtaskStatus: [
        {
          text: 'Input Node',
          color: '#3281A8',
          legendDotXLoc: '10',
          legendDotYLoc: '20',
          legendTextXLoc: '20',
          legendTextYLoc: '21',
        },
        {
          text: 'Output Node',
          color: '#C334EB',
          legendDotXLoc: '135',
          legendDotYLoc: '20',
          legendTextXLoc: '145',
          legendTextYLoc: '21',
        },
      ],
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
    svgContainer
      .append('circle')
      .attr('cx', dotX)
      .attr('cy', dotY)
      .attr('r', 6)
      .style('fill', color);

    svgContainer
      .append('text')
      .attr('x', textX)
      .attr('y', textY)
      .text(text)
      .style('font-size', '15px')
      .attr('alignment-baseline', 'middle');
  }

  fetchGraphDetail() {
    const { sessionId, taskId, tileableId } = this.props;

    if (sessionId === undefined || taskId === undefined || tileableId === undefined) {
      return;
    }

    fetch(`api/session/${sessionId}/task/${taskId
    }/${tileableId}/subtask?with_input_output=true`)
      .then(res => res.json())
      .then((res) => {
        let subtaskList = [];
        let dependencyList = [];

        if (Object.keys(res).length > 0) {
          subtaskList = Object.keys(res).map((subtaskId) => {
            return (
              {
                id: subtaskId,
                name: res[subtaskId].name
              }
            );
          });

          Object.keys(res).filter(
            subtaskId => res[subtaskId].fromSubtaskIds.length > 0
          ).forEach((subtaskId) => {
            let fromNodeIds = res[subtaskId].fromSubtaskIds;

            fromNodeIds.forEach((fromNodeId) => {
              dependencyList.push(
                {
                  fromNodeId,
                  toNodeId: subtaskId,
                }
              );
            });
          });
        }

        this.setState({
          subtasks: subtaskList,
          dependencies: dependencyList,
        });
      });
  }

  fetchSubtaskDetail() {
    const { sessionId, taskId, tileableId } = this.props;

    if (sessionId === undefined || taskId === undefined || tileableId === undefined) {
      return;
    }

    fetch(`api/session/${sessionId}/task/${taskId
    }/${tileableId}/subtask?with_input_output=true`)
      .then(res => res.json())
      .then((res) => {
        this.setState({
          subtaskDetails: res
        });
      });
  }

  componentDidMount() {
    if (this.interval !== undefined) {
      clearInterval(this.interval);
    }
    this.interval = setInterval(() => this.fetchSubtaskDetail(), 1000);
    this.fetchSubtaskDetail();
    this.fetchGraphDetail();

    // Create the legend for DAG
    const legendSVG = d3Select('#subtasks-legend');
    this.state.subtaskStatus.forEach((status) => this.generateGraphLegendItem(
      legendSVG,
      status.legendDotXLoc,
      status.legendDotYLoc,
      status.legendTextXLoc,
      status.legendTextYLoc,
      status.color,
      status.text
    ));
  }

  /* eslint no-unused-vars: ["error", { "args": "none" }] */
  componentDidUpdate(prevProps, prevStates, snapshot) {
    if (prevProps.tileableId !== this.props.tileableId) {
      this.fetchSubtaskDetail();
      this.fetchGraphDetail();
    }
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  render() {
    const dagStyle = {
      margin: 15,
      width: '90%',
      height: '80%',
      minHeight: 200,
    };

    if (this.state === undefined ||
        this.state.subtasks === undefined ||
        this.state.dependencies === undefined ||
        this.state.subtaskDetails === undefined) {
      return (
        <div>Loading</div>
      );
    }

    return (
      <React.Fragment>
        <h2>Subtask Graph Info:</h2>
        {
          this.state.subtasks.length + this.state.dependencies.length > 2000 &&
            <div>
              Warning: this subtask graph contains a lot of elements and may take some time to load
            </div>
        }
        <svg
          id='subtasks-legend'
          style={{ marginLeft: '6%', width: '90%', height: 50 }}
        />
        <DAGCanvasChart
          graphName='subtaskGraph'
          dagStyle={dagStyle}
          nodes={this.state.subtasks}
          nodeShape='circle'
          nodesStatus={this.state.subtaskDetails}
          dependencies={this.state.dependencies}
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
