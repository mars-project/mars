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
import sum from 'lodash/sum';
import uniq from 'lodash/uniq';
import without from 'lodash/without';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import PropTypes from 'prop-types';
import { useStyles } from './Style';
import Title from './Title';
import { toReadableSize } from './Utils';

class NodeInfo extends React.Component {
  constructor(props) {
    super(props);
    this.nodeRole = props.nodeRole.toLowerCase();
    this.state = {};
  }

  refreshInfo() {
    const roleId = (this.nodeRole === 'supervisor' ? 0 : 1);
    fetch(`api/cluster/nodes?role=${roleId}&env=1&resource=1&exclude_statuses=-1`)
      .then((res) => res.json())
      .then((res) => {
        const { state } = this;
        state[this.nodeRole] = res.nodes;
        this.setState(state);
      });
  }

  componentDidMount() {
    this.interval = setInterval(() => this.refreshInfo(), 5000);
    this.refreshInfo();
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  render() {
    if (this.state === undefined || this.state[this.nodeRole] === undefined) {
      return (
        <div>Loading</div>
      );
    }

    const roleData = this.state[this.nodeRole];

    const gatherResourceStats = (prop) => sum(
      Object.values(roleData).map(
        (val) => sum(Object.values(val.resource).map((a) => a[prop])),
      ),
    );

    const resourceStats = {
      cpu_total: gatherResourceStats('cpu_total'),
      cpu_avail: gatherResourceStats('cpu_avail'),
      memory_total: gatherResourceStats('memory_total'),
      memory_avail: gatherResourceStats('memory_avail'),
      git_branches: uniq(without(Object.values(roleData).map(
        (val) => {
          const { git_info } = val.env;
          return git_info === undefined ? undefined : (`${git_info.hash} ${git_info.ref}`);
        },
      ), undefined)),
    };
    resourceStats.cpu_used = resourceStats.cpu_total - resourceStats.cpu_avail;
    resourceStats.memory_used = resourceStats.memory_total - resourceStats.memory_avail;

    return (
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell style={{ fontWeight: 'bolder' }}>Item</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>Value</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell>Count</TableCell>
            <TableCell>{Object.keys(this.state[this.nodeRole]).length}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>CPU Info</TableCell>
            <TableCell>
              <div>
                Usage:
                {resourceStats.cpu_used.toFixed(2)}
              </div>
              <div>
                Total:
                {resourceStats.cpu_total.toFixed(2)}
              </div>
            </TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Memory Info</TableCell>
            <TableCell>
              <div>
                Usage:
                {toReadableSize(resourceStats.memory_used)}
              </div>
              <div>
                Total:
                {toReadableSize(resourceStats.memory_total)}
              </div>
            </TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Git Branch</TableCell>
            <TableCell>
              {resourceStats.git_branches.map((branch, idx) => (
                <div key={`${this.nodeRole}_git_branch_${idx.toString()}`}>{branch}</div>
              ))}
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
  }
}

NodeInfo.propTypes = {
  nodeRole: PropTypes.string,
};

export default function Dashboard() {
  const classes = useStyles();
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Title>Dashboard</Title>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <Title>Supervisors</Title>
          <NodeInfo nodeRole="supervisor" />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <Title>Workers</Title>
          <NodeInfo nodeRole="worker" />
        </Paper>
      </Grid>
    </Grid>
  );
}
