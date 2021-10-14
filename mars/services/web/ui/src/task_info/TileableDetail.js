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
import PropTypes from 'prop-types';
import { Tabs, Tab } from '@material-ui/core';
const SubtaskGraph = lazy(() => {
    return import('./SubtaskGraph');
});


class TileableDetail extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            displayedTileableDetail: 0,
        };

        this.handleDetailTabChange = this.handleDetailTabChange.bind(this);
    }

    handleDetailTabChange(e, newDetailKey) {
        this.setState({
            displayedTileableDetail: newDetailKey
        });
    }

    render() {
        if (this.props === undefined) {
            return null;
        }

        console.log(this.props.tileable);

        return (
            this.props.tileable
                ?
                <React.Fragment>
                    <Tabs value={this.state.displayedTileableDetail} onChange={this.handleDetailTabChange}>
                        <Tab label='Tileable Info' />
                        <Tab label='Subtask Info' />
                    </Tabs><br />

                    <div>
                        {
                            this.state.displayedTileableDetail === 0
                                ?
                                <React.Fragment>
                                    <h2>Tileable Graph Info:</h2>
                                    <div>Tileable ID: <br/>{this.props.tileable.id}</div><br/>
                                    <div>Tileable Name: <br/>{this.props.tileable.name}</div><br/><br />

                                    <div>Tileable Properties:</div><br/>
                                    <div style={{ maxHeight: '50%', overflow: 'auto' }}>
                                        {
                                            Object.keys(this.props.tileable.properties).map((key) => {
                                                return (
                                                    <div key={key}>
                                                        {key}: {this.props.tileable.properties[key]}
                                                        <br />
                                                    </div>
                                                )
                                            })
                                        }
                                    </div>
                                </React.Fragment>
                                :
                                <Suspense fallback={<div>Loading...</div>}>
                                    <SubtaskGraph
                                        sessionId={this.props.sessionId}
                                        taskId={this.props.taskId}
                                        tileableId={this.props.tileable.id}
                                    />
                                </Suspense>
                        }
                    </div>
                </React.Fragment>
                :
                <React.Fragment>
                    Select a tileable to view its detail
                </React.Fragment>
        );
    }
}

TileableDetail.propTypes = {
    tileable: PropTypes.shape({
        id: PropTypes.string,
        name: PropTypes.string,
    }),
    sessionId: PropTypes.string.isRequired,
    taskId: PropTypes.string.isRequired,
};

export default TileableDetail;
