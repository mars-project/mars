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
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableBody from '@material-ui/core/TableBody';
import PropTypes from 'prop-types';
import Title from '../Title';
import {useStyles} from '../Style';
import {formatTime, getTaskStatusText} from '../Utils';
import { Link } from 'react-router-dom';


class TaskList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  refreshInfo() {
    fetch(`api/session/${this.props.sessionId}/task?progress=1`)
      .then((res) => res.json())
      .then((res) => {
        this.setState(res);
      });
  }

  componentDidMount() {
    if (this.interval !== undefined) {
      clearInterval(this.interval);
    }
    this.interval = setInterval(() => this.refreshInfo(), 5000);
    this.refreshInfo();
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  formatTaskStatus(task) {
    let status = getTaskStatusText(task.status);
    if (status === 'terminated') {
      status = task.error ? 'failed' : 'succeeded';
    }
    return status;
  }

  render() {
    if (this.state === undefined || this.state.tasks === undefined) {
      return (
        <div>Loading</div>
      );
    }
    return (
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell style={{ fontWeight: 'bolder' }}>Task ID</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>Start Time</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>End Time</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>Progress</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {this.state.tasks.map((task) => (
            <TableRow key={`task_row_${task.task_id}`}>
              <TableCell>
                <Link to={`/session/${this.props.sessionId}/task/${task.task_id}`}>
                  {task.task_id}
                </Link>
              </TableCell>
              <TableCell>{formatTime(task.start_time)}</TableCell>
              <TableCell>{task.end_time ? formatTime(task.end_time) : 'N/A'}</TableCell>
              <TableCell>{`${Math.floor(task.progress * 100).toString()}%`}</TableCell>
              <TableCell>{this.formatTaskStatus(task)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }
}

TaskList.propTypes = {
  sessionId: PropTypes.string,
};

export default function TaskListPage(props) {
  const classes = useStyles();
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Title>
                    Session {props.sessionId}
        </Title>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <TaskList sessionId={props.sessionId} />
        </Paper>
      </Grid>
    </Grid>
  );
}

TaskListPage.propTypes = {
  sessionId: PropTypes.string,
};
