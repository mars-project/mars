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
import DAGChart from './charts/DAGChart';


export default class SubtaskGraph extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            subtasks: [],
            dependencies: [],
            subtaskDetails: {},
        };
    }

    fetchGraphDetail() {
        const { sessionId, taskId, tileableId } = this.props;

        if (sessionId === undefined || taskId === undefined || tileableId === undefined) {
            return;
        }

        fetch(`api/session/${sessionId}/task/${taskId
        }/${tileableId}/subtask_graph`)
            .then(res => res.json())
            .then((res) => {
                this.setState({
                    subtasks: res.subtasks.map(subtask => {
                        return (
                            {
                                id: subtask.subtaskId,
                                name: subtask.subtaskName,
                            }
                        )
                    }),

                    dependencies: res.dependencies.map(({fromSubtaskId, toSubtaskId}) => {
                        return (
                            {
                                fromNodeId: fromSubtaskId,
                                toNodeId: toSubtaskId,
                            }
                        )
                    }),
                });
            });
    }

    fetchSubtaskDetail() {
        const { sessionId, taskId, tileableId } = this.props;

        if (sessionId === undefined || taskId === undefined || tileableId === undefined) {
            return;
        }

        fetch(`api/session/${sessionId}/task/${taskId
        }/${tileableId}/subtask_detail`)
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
        this.interval = setInterval(() => this.fetchSubtaskDetail(), 5000);
        this.fetchSubtaskDetail();
        this.fetchGraphDetail();
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
            <DAGChart
                graphName='subtaskGraph'
                dagStyle={dagStyle}
                nodes={this.state.subtasks}
                nodeShape='circle'
                nodesStatus={this.state.subtaskDetails}
                dependencies={this.state.dependencies}
            />
        );
    }
}

SubtaskGraph.propTypes = {
    sessionId: PropTypes.string.isRequired,
    taskId: PropTypes.string.isRequired,
    tileableId: PropTypes.string.isRequired,
};
