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
import MenuItem from '@material-ui/core/MenuItem';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Select from '@material-ui/core/Select';
import Switch from '@material-ui/core/Switch';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import PropTypes from 'prop-types';
import Title from '../Title';
import {formatTime} from '../Utils';


export default class NodeStackTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loaded: false,
    };
    this.interval = undefined;
  }

  componentDidMount() {
    this.loadProcesses();
    this.refreshStack();
  }

  loadProcesses() {
    fetch(`api/cluster/pools?address=${this.props.endpoint}`)
      .then((res) => res.json())
      .then((res) => {
        const pools = res.pools;
        const {state} = this;
        let poolToName = [];

        for (let i = 0; i < pools.length; i++) {
          let label = pools[i].label;
          if (label)
            poolToName.push(`${i}: ${pools[i].label}`);
          else
            poolToName.push(`${i}`);
        }
        state.loaded = true;
        state.selectedIndex = 0;
        state.pools = poolToName;
        this.setState(state);
      });
  }

  refreshStack() {
    fetch(`api/cluster/stacks?address=${this.props.endpoint}`)
      .then((res) => res.json())
      .then((res) => {
        const stacks = res.stacks;
        const {state} = this;
        let resultStacks = [];

        for (let i = 0; i < stacks.length; i++) {
          const inStacks = stacks[i];
          let stackObj = {};
          if (!inStacks) {
            resultStacks.push(undefined);
            continue;
          }
          Object.keys(inStacks).forEach((threadKey) => {
            stackObj[threadKey] = join(inStacks[threadKey], '');
          });
          resultStacks.push(stackObj);
        }
        state.loaded = true;
        state.generateTime = res.generate_time;
        state.stacks = resultStacks;
        this.setState(state);
      });
  }

  renderStack() {
    if (!this.state.stacks || !this.state.stacks[this.state.selectedIndex])
      return <React.Fragment />;
    const stacksObj = this.state.stacks[this.state.selectedIndex];
    return (
      <div>
        <Title component="h3">Generate Time: {formatTime(this.state.generateTime)}</Title>
        {Object.keys(stacksObj).map((threadKey) => (
          <div key={`block-${threadKey}`}>
            <Title component="h3">{threadKey}</Title>
            <Paper style={{width: '100%', overflow: 'auto'}}>
              <pre style={{fontSize: 'smaller'}}>{stacksObj[threadKey]}</pre>
            </Paper>
          </div>
        ))}
      </div>
    );
  }

  render() {
    if (!this.state.loaded) {
      return (
        <div>Loading</div>
      );
    }

    const onSelectProcess = (event) => {
      const {state} = this;
      console.log(event);
      state.selectedIndex = parseInt(event.target.value);
      this.setState(state);
    };

    const onRefreshChange = (event) => {
      if (event.target.checked) {
        this.interval = setInterval(() => this.refreshStack(), 5000);
        this.refreshStack();
      } else {
        clearInterval(this.interval);
      }
    };

    return (
      <div>
        <div style={{display: 'flex'}}>
          <FormControl fullWidth>
            <InputLabel id="process-index-select-label">Process</InputLabel>
            <Select
              labelId="process-index-select-label"
              id="process-index-select"
              label="Process"
              style={{width: '100%'}}
              value={`${this.state.selectedIndex}`}
              onChange={onSelectProcess}
            >
              {this.state.pools.map((s, idx) => (
                <MenuItem key={`item-${idx}`} value={idx}>{s}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl>
            <FormControlLabel
              style={{display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-end',
              }}
              control={
                <Switch onChange={onRefreshChange} />
              }
              label={<Typography style={{fontSize: 'smaller'}}>Refresh</Typography>}
              labelPlacement="top"
            />
          </FormControl>
        </div>
        <div>
          {this.renderStack()}
        </div>
      </div>
    );
  }
}

NodeStackTab.propTypes = {
  endpoint: PropTypes.string,
};
