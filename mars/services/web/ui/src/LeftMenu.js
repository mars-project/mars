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
import Divider from '@material-ui/core/Divider';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import DashboardIcon from '@material-ui/icons/Dashboard';
import SupervisedUserCircleIcon from '@material-ui/icons/SupervisedUserCircle';
import MemoryIcon from '@material-ui/icons/Memory';
import AssignmentReturnedIcon from '@material-ui/icons/AssignmentReturned';
import DescriptionIcon from '@material-ui/icons/Description';
import GitHub from '@material-ui/icons/GitHub';
import { useStyles } from './Style';

export default function LeftMenu() {
  const classes = useStyles();
  const getHashPath = () => (window.location.hash.substring(1));
  const [hash, setHash] = React.useState(getHashPath());

  window.addEventListener('hashchange', () => {
    setHash(getHashPath());
  }, false);

  const genNodeSubMenu = (nodeRole) => {
    const match = hash.match(/^\/(supervisor|worker)\/([^/]+)/, 1);
    return (
      match && nodeRole === match[1] &&
        <React.Fragment>
          <Divider />
          <List component="div" disablePadding>
            <ListItem button className={classes.nestedListItem}
              component={Link} to={`/${match[1]}/${match[2]}`}
              selected={true}
            >
              <ListItemIcon />
              <ListItemText primary={match[2]} />
            </ListItem>
          </List>
        </React.Fragment>
    );
  };

  const genSessionSubMenu = () => {
    const match = hash.match(/^\/session\/([^/]+)\/task/, 1);
    return (
      match &&
        <React.Fragment>
          <Divider />
          <List component="div" disablePadding>
            <ListItem button className={classes.nestedListItem}
              component={Link} to={`/session/${match[1]}/task`}
              selected={true}
            >
              <ListItemIcon />
              <ListItemText primary={match[1]} />
            </ListItem>
          </List>
        </React.Fragment>
    );
  };

  return (
    <List className={classes.leftMenu}>
      <div>
        <ListItem button component={Link} to="/" selected={hash === '/'}>
          <ListItemIcon>
            <DashboardIcon />
          </ListItemIcon>
          <ListItemText primary="Dashboard" />
        </ListItem>
        <ListItem button component={Link} to="/supervisor"
          selected={hash.startsWith('/supervisor')}
        >
          <ListItemIcon>
            <SupervisedUserCircleIcon />
          </ListItemIcon>
          <ListItemText primary="Supervisors" />
        </ListItem>
        {genNodeSubMenu('supervisor')}
        <ListItem button component={Link} to="/worker"
          selected={hash.startsWith('/worker')}
        >
          <ListItemIcon>
            <MemoryIcon />
          </ListItemIcon>
          <ListItemText primary="Workers" />
        </ListItem>
        {genNodeSubMenu('worker')}
        <ListItem button component={Link} to="/session"
          selected={hash === '/session'}
        >
          <ListItemIcon>
            <AssignmentReturnedIcon />
          </ListItemIcon>
          <ListItemText primary="Sessions" />
        </ListItem>
        {genSessionSubMenu()}
        <ListItem button component="a" href="https://docs.pymars.org" target="_blank">
          <ListItemIcon>
            <DescriptionIcon />
          </ListItemIcon>
          <ListItemText primary="Documentation" />
        </ListItem>
        <ListItem button component="a" href="https://github.com/mars-project/mars" target="_blank">
          <ListItemIcon>
            <GitHub />
          </ListItemIcon>
          <ListItemText primary="Repository" />
        </ListItem>
      </div>
    </List>
  );
}
