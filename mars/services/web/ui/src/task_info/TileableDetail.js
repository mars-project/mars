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
                <div>
                    <div>Tileable ID: <br/>{this.props.tileable.tileable_id}</div><br/>
                    <div>Tileable Name: <br/>{this.props.tileable.tileable_name}</div><br/>
                </div>
                :
                <div>
                    Select a tileable to view its detail
                </div>
        );
    }
}

TileableDetail.propTypes = {
    tileable: PropTypes.shape({
        tileable_id: PropTypes.string,
        tileable_name: PropTypes.string,
    }),
};

export default TileableDetail;
