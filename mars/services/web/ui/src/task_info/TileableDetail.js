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
const SubtaskGraph = lazy(() => {
    return import('./SubtaskGraph');
});

class TileableDetail extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    render() {
        if (this.props === undefined) {
            return null;
        }

        return (
            this.props.tileable
                ?
                <React.Fragment>
                    <h2>Tileable Graph Info:</h2>
                    <div>Tileable ID: <br/>{this.props.tileable.id}</div><br/>
                    <div>Tileable Name: <br/>{this.props.tileable.name}</div><br/><br />
                    <Suspense fallback={<div>Loading...</div>}>
                        <SubtaskGraph
                            sessionId={this.props.sessionId}
                            taskId={this.props.taskId}
                            tileableId={this.props.tileable.id}
                        />
                    </Suspense>
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
