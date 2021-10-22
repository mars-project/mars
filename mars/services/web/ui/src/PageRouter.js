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

import React, { lazy, Suspense } from 'react';
import {
  Switch,
  Route,
  useParams,
} from 'react-router-dom';
import PropTypes from 'prop-types';
import Dashboard from './Dashboard';
import NodeListPage from './node_info/NodeListPage';
import SupervisorDetailPage from './node_info/SupervisorDetailPage';
import WorkerDetailPage from './node_info/WorkerDetailPage';
import SessionListPage from './SessionListPage';
import TaskListPage from './task_info/TaskListPage';
const TaskDetail = lazy(() => {
  return import('./task_info/TaskDetail');
});

function NodePageWrapper(props) {
  const { endpoint } = useParams();
  const { nodeRole, component, ...other } = props;
  const ComponentTag = component;

  return (
    <ComponentTag endpoint={endpoint} nodeRole={nodeRole} {...other} />
  );
}

NodePageWrapper.propTypes = {
  nodeRole: PropTypes.string,
  component: PropTypes.elementType,
};

function TaskPageWrapper() {
  const { session_id } = useParams();
  return (
    <TaskListPage sessionId={session_id} />
  );
}

export default function PageRouter() {
  return (
    <Switch>
      <Route exact path="/supervisor" render={() => (<NodeListPage nodeRole="supervisor" />)} />
      <Route exact path="/worker" render={() => (<NodeListPage nodeRole="worker" />)} />
      <Route exact path="/session" render={() => (<SessionListPage />)} />
      <Route
        exact
        path="/supervisor/:endpoint"
        render={() => (<NodePageWrapper component={SupervisorDetailPage} nodeRole="supervisor" />)}
      />
      <Route
        exact
        path="/worker/:endpoint"
        render={() => (<NodePageWrapper component={WorkerDetailPage} nodeRole="worker" />)}
      />
      <Route exact path="/session/:session_id/task" render={() => (<TaskPageWrapper />)} />
      <Route exact path="/" component={Dashboard} />
      <Suspense fallback={<div>Loading...</div>}>
        <Route exact path="/session/:session_id/task/:task_id" component={TaskDetail} />
      </Suspense>
    </Switch>
  );
}
