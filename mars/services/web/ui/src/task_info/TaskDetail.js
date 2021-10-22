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
import { withRouter } from 'react-router-dom';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Divider from '@material-ui/core/Divider';
import Title from '../Title';
import TaskTileableGraph from './TaskTileableGraph';
import TileableDetail from './TileableDetail';


class TaskDetail extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedTileable: null,
    };
  }

  render() {
    if (this.props === undefined) {
      return null;
    }
    const tileableClick = (e, tileable) => {
      this.setState({
        selectedTileable: tileable
      });
    };
    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Title>Task {this.props.match.params.task_id}</Title>
        </Grid>
        <Grid item xs={12}>
          <Paper>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TaskTileableGraph
                  sessionId={this.props.match.params.session_id}
                  taskId={this.props.match.params.task_id}
                  onTileableClick={tileableClick}
                />
              </Grid>
              <Divider
                orientation='vertical' flexItem
                style={{marginRight:'-1px'}}
              />
              <Grid item xs={12} sm={6}>
                <Grid item xs={12}>
                  <TileableDetail
                    tileable={this.state.selectedTileable}
                    sessionId={this.props.match.params.session_id}
                    taskId={this.props.match.params.task_id}
                  />
                </Grid>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  }
}

TaskDetail.propTypes = {
  match: PropTypes.shape({
    params: PropTypes.shape({
      session_id: PropTypes.string.isRequired,
      task_id: PropTypes.string.isRequired,
    })
  }),
};

export default withRouter(TaskDetail);
