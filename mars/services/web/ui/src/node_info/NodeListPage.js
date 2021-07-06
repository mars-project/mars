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
import {Link} from "react-router-dom";
import sum from "lodash/sum";
import Table from "@material-ui/core/Table";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import TableBody from "@material-ui/core/TableBody";
import {toReadableSize} from "../Utils";
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import { useStyles } from "../Style";
import Title from "../Title";
import {formatTime} from "../Utils";

class NodeList extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    refreshInfo() {
        let roleId = (this.nodeRole === 'supervisor' ? 0 : 1);
        fetch('api/cluster/nodes?role=' + roleId + '&resource=1')
            .then(res => res.json())
            .then((res) => {
                let state = this.state;
                state[this.nodeRole] = res;
                this.setState(state);
            })
    }

    reloadComponent() {
        this.nodeRole = this.props.nodeRole.toLowerCase();

        if (this.interval !== undefined)
            clearInterval(this.interval);
        this.interval = setInterval(() => this.refreshInfo(), 5000);
        this.refreshInfo();
    }

    componentDidMount() {
        this.reloadComponent();
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (this.props.nodeRole !== prevProps.nodeRole) {
            this.reloadComponent();
        }
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

        const calcNodeStats = (nodeData, prop) => (
            sum(Object.values(nodeData['resource']).map((a) => a[prop]))
        );

        const calcNodeStatGroup = (nodeData, prefix, reprFun) => {
            let avail = calcNodeStats(nodeData, prefix + '_avail');
            let total = calcNodeStats(nodeData, prefix + '_total');
            return reprFun(total - avail) + ' / ' + reprFun(total);
        };

        const roleData = this.state[this.nodeRole];

        return (
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell style={{fontWeight: 'bolder'}}>Endpoint</TableCell>
                        <TableCell style={{fontWeight: 'bolder'}}>CPU</TableCell>
                        <TableCell style={{fontWeight: 'bolder'}}>Memory</TableCell>
                        <TableCell style={{fontWeight: 'bolder'}}>Update Time</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {
                        Object.keys(roleData).map((endpoint) => (
                            <TableRow key={'nodeList_' + this.nodeRole + '_' + endpoint}>
                                <TableCell><Link to={"/" + this.nodeRole + "/" + endpoint}>{endpoint}</Link></TableCell>
                                <TableCell>{calcNodeStatGroup(roleData[endpoint], 'cpu', (v) => v.toFixed(2))}</TableCell>
                                <TableCell>{calcNodeStatGroup(roleData[endpoint], 'memory', toReadableSize)}</TableCell>
                                <TableCell>{formatTime(roleData[endpoint]['update_time'])}</TableCell>
                            </TableRow>
                        ))
                    }
                </TableBody>
            </Table>
        );
    }
}

export default function NodeListPage(props) {
    const classes = useStyles();
    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Title>{props.nodeRole.replace(/\w/, first => first.toUpperCase()) + 's'}</Title>
            </Grid>
            <Grid item xs={12}>
                <Paper className={classes.paper}>
                    <React.Fragment>
                        <NodeList nodeRole={props.nodeRole} />
                    </React.Fragment>
                </Paper>
            </Grid>
        </Grid>
    )
}
