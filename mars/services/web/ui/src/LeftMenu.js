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
import { Link } from "react-router-dom";
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import DashboardIcon from '@material-ui/icons/Dashboard';
import SupervisedUserCircleIcon from '@material-ui/icons/SupervisedUserCircle';
import MemoryIcon from '@material-ui/icons/Memory';
import AssignmentReturnedIcon from '@material-ui/icons/AssignmentReturned';
import AssignmentIcon from '@material-ui/icons/Assignment';
import DescriptionIcon from '@material-ui/icons/Description';
import GitHub from "@material-ui/icons/GitHub";

export default function LeftMenu() {
  return (
      <List>
        <div>
          <ListItem button component={Link} to="/">
            <ListItemIcon>
              <DashboardIcon/>
            </ListItemIcon>
            <ListItemText primary="Dashboard"/>
          </ListItem>
          <ListItem button component={Link} to="/supervisor">
            <ListItemIcon>
              <SupervisedUserCircleIcon/>
            </ListItemIcon>
            <ListItemText primary="Supervisors"/>
          </ListItem>
          <ListItem button component={Link} to="/worker">
            <ListItemIcon>
              <MemoryIcon/>
            </ListItemIcon>
            <ListItemText primary="Workers"/>
          </ListItem>
          <ListItem button component="a" href="/#/session">
            <ListItemIcon>
              <AssignmentReturnedIcon/>
            </ListItemIcon>
            <ListItemText primary="Sessions"/>
          </ListItem>
          <ListItem button component="a" href="/#/tasks">
            <ListItemIcon>
              <AssignmentIcon/>
            </ListItemIcon>
            <ListItemText primary="Task Details"/>
          </ListItem>
          <ListItem button component="a" href="https://docs.pymars.org" target="_blank">
            <ListItemIcon>
              <DescriptionIcon/>
            </ListItemIcon>
            <ListItemText primary="Documentation"/>
          </ListItem>
          <ListItem button component="a" href="https://github.com/mars-project/mars" target="_blank">
            <ListItemIcon>
              <GitHub/>
            </ListItemIcon>
            <ListItemText primary="Repository"/>
          </ListItem>
        </div>
      </List>
  )
}
