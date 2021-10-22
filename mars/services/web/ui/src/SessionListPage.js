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
import { Link } from 'react-router-dom';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableBody from '@material-ui/core/TableBody';
import { useStyles } from './Style';
import Title from './Title';

class SessionList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  refreshInfo() {
    fetch('api/session')
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

  render() {
    if (this.state === undefined || this.state.sessions === undefined) {
      return (
        <div>Loading</div>
      );
    }

    return (
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell style={{ fontWeight: 'bolder' }}>Session ID</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {this.state.sessions.map((session) => (
            <TableRow key={`session_row_${session.session_id}`}>
              <TableCell>
                <Link to={`/session/${session.session_id}/task`}>{session.session_id}</Link>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }
}

export default function SessionListPage() {
  const classes = useStyles();
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Title>Sessions</Title>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <SessionList />
        </Paper>
      </Grid>
    </Grid>
  );
}
