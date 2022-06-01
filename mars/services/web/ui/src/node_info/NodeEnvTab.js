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

import join from 'lodash/join';
import React from 'react';
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableBody from '@material-ui/core/TableBody';
import PropTypes from 'prop-types';


export default class NodeEnvTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loaded: false,
    };
  }

  componentDidMount() {
    fetch(`api/cluster/nodes?nodes=${this.props.endpoint
    }&env=1&exclude_statuses=-1`)
      .then((res) => res.json())
      .then((res) => {
        const state = res.nodes[this.props.endpoint].env;
        state.loaded = true;
        this.setState(state);
      });
  }

  render() {
    if (!this.state.loaded) {
      return (
        <div>Loading</div>
      );
    }
    return (
      <Table>
        <TableHead>
          <TableRow>
            <TableCell style={{ fontWeight: 'bolder' }}>Item</TableCell>
            <TableCell style={{ fontWeight: 'bolder' }}>Value</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell>Endpoint</TableCell>
            <TableCell>{this.props.endpoint}</TableCell>
          </TableRow>
          {Boolean(this.state.k8s_pod_name) &&
            <TableRow>
              <TableCell>Kubernetes Pod</TableCell>
              <TableCell>{this.state.k8s_pod_name}</TableCell>
            </TableRow>
          }
          {Boolean(this.state.yarn_container_id) &&
            <TableRow>
              <TableCell>Yarn Container ID</TableCell>
              <TableCell>{this.state.yarn_container_id}</TableCell>
            </TableRow>
          }
          <TableRow>
            <TableCell>Platform</TableCell>
            <TableCell>{this.state.platform}</TableCell>
          </TableRow>
          {Boolean(this.state.cuda_info) &&
            <TableRow>
              <TableCell>CUDA</TableCell>
              <TableCell>
                <div key='cuda'>
                  CUDA Version: {this.state.cuda_info.cuda}
                </div>
                <div key='driver'>
                  Driver Version: {this.state.cuda_info.driver}
                </div>
                <div key='products'>
                  Devices: {join(this.state.cuda_info.products, ',')}
                </div>
              </TableCell>
            </TableRow>
          }
          <TableRow>
            <TableCell>Git Branch</TableCell>
            <TableCell>{`${this.state.git_info.hash} ${this.state.git_info.ref}`}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Command</TableCell>
            <TableCell>{join(this.state.command_line, ' ')}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Python Version</TableCell>
            <TableCell>{this.state.python_version}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Package Versions</TableCell>
            <TableCell>
              {Object.keys(this.state.package_versions).map((key) => (
                <div key={`package_${key}@${this.props.endpoint}`}>
                  {`${key}: ${this.state.package_versions[key]}`}
                </div>
              ))}
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
  }
}

NodeEnvTab.propTypes = {
  endpoint: PropTypes.string,
};
