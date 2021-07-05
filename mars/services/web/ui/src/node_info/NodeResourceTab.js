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
import Table from "@material-ui/core/Table";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import TableBody from "@material-ui/core/TableBody";
import Title from "../Title";
import {OptionalElement, toReadableSize} from "../Utils";

export default class NodeResourceTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            loaded: false,
        };
    }

    refreshInfo() {
        fetch('api/cluster/nodes?nodes=' + this.props.endpoint + '&resource=1&state=1')
            .then(res => res.json())
            .then((res) => {
                let result = res[this.props.endpoint];
                let state = this.state;
                state['loaded'] = true;
                state['resource'] = result.resource;
                state['state'] = result.state;
                this.setState(state);
            })
    }

    componentDidMount() {
        this.interval = setInterval(() => this.refreshInfo(), 5000);
        this.refreshInfo();
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    generateBandRows() {
        let rows = [];
        Object.keys(this.state.resource).map((band) => {
            let band_rows = [];
            const band_res = this.state.resource[band];
            if (band_res['cpu_avail'] !== undefined) {
                band_rows.push([(<TableCell key={band + "-cpu-item"}>CPU</TableCell>), (
                    <TableCell key={band + "-cpu-value"}>
                        <div>Usage: {(band_res.cpu_total - band_res.cpu_avail).toFixed(2)}</div>
                        <div>Total: {band_res.cpu_total.toFixed(2)}</div>
                    </TableCell>
                )])
            }
            if (band_res['memory_avail'] !== undefined) {
                band_rows.push([(<TableCell key={band + "-memory-item"}>Memory</TableCell>), (
                    <TableCell key={band + "-memory-value"}>
                        <div>Usage: {toReadableSize(band_res.memory_total - band_res.memory_avail)}</div>
                        <div>Total: {toReadableSize(band_res.memory_total)}</div>
                    </TableCell>
                )])
            }
            if (band_rows.length > 0) {
                rows.push((
                    <TableRow>
                        <TableCell key={band + "-band"} colSpan={2}>{band}</TableCell>
                    </TableRow>
                ))
                band_rows.map((cells, idx) => {
                    rows.push((<TableRow key={band + '-' + idx.toString()}>{cells}</TableRow>));
                });
            }
        });
        return rows
    }

    generateDiskRows() {
        let rows = [];
        if (this.state.state['disk']['partitions'] === undefined)
            return null;

        Object.keys(this.state.state['disk']['partitions']).map((path) => {
            const part_desc = this.state.state['disk']['partitions'][path];
            rows = rows.concat([
                (
                    <TableRow key={path}>
                        <TableCell colSpan={2}>{path}</TableCell>
                    </TableRow>
                ),
                (
                    <TableRow>
                        <TableCell key={path + "-size-key"}>Size</TableCell>
                        <TableCell key={path + "-size-value"}>
                            <div>Usage: {toReadableSize(part_desc['size_used'])}</div>
                            <div>Total: {toReadableSize(part_desc['size_total'])}</div>
                        </TableCell>
                    </TableRow>
                )
            ]);
            if (part_desc['inode_used'] !== undefined) {
                rows.push((
                    <TableRow>
                        <TableCell key={path + "-inode-key"}>INode</TableCell>
                        <TableCell key={path + "-inode-value"}>
                            <div>Usage: {toReadableSize(part_desc['inode_used'])}</div>
                            <div>Total: {toReadableSize(part_desc['inode_total'])}</div>
                        </TableCell>
                    </TableRow>
                ));
            }
            if (part_desc['reads'] !== undefined) {
                rows.push((
                    <TableRow>
                        <TableCell key={path + "-io-key"}>IO</TableCell>
                        <TableCell key={path + "-io-value"}>
                            <div>Reads: {toReadableSize(part_desc['reads'])}</div>
                            <div>Writes: {toReadableSize(part_desc['writes'])}</div>
                        </TableCell>
                    </TableRow>
                ));
            }
        });
        return rows;
    }

    render() {
        if (!this.state.loaded) {
            return (
                <div>Loading</div>
            );
        }
        if (!this.state.loaded) {
            return (
                <div>Loading</div>
            );
        }
        return (
            <div>
                <Title component={"h3"}>Bands</Title>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell style={{fontWeight: 'bolder'}}>Item</TableCell>
                            <TableCell style={{fontWeight: 'bolder'}}>Value</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {this.generateBandRows()}
                    </TableBody>
                </Table>
                <Title component={"h3"}>IO Summary</Title>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell style={{fontWeight: 'bolder'}}>Item</TableCell>
                            <TableCell style={{fontWeight: 'bolder'}}>Value</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <OptionalElement condition={this.state.state['iowait']}>
                            <TableRow>
                                <TableCell>IO Wait</TableCell>
                                <TableCell>{this.state.state['iowait']}</TableCell>
                            </TableRow>
                        </OptionalElement>
                        <TableRow>
                            <TableCell>Disk</TableCell>
                            <TableCell>
                                <div>Reads: {this.state.state['disk']['reads']}</div>
                                <div>Writes: {this.state.state['disk']['writes']}</div>
                            </TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell>Network</TableCell>
                            <TableCell>
                                <div>Receives: {this.state.state['network']['receives']}</div>
                                <div>Sends: {this.state.state['network']['sends']}</div>
                            </TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
                <OptionalElement condition={this.state.state['disk']['partitions']}>
                    <Title component={"h3"}>Disks</Title>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell style={{fontWeight: 'bolder'}}>Item</TableCell>
                                <TableCell style={{fontWeight: 'bolder'}}>Value</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {this.generateDiskRows()}
                        </TableBody>
                    </Table>
                </OptionalElement>
                <OptionalElement condition={this.state.state['quota']}>
                    <Title component={"h3"}>Quota</Title>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell style={{fontWeight: 'bolder'}}>Band</TableCell>
                                <TableCell style={{fontWeight: 'bolder'}}>Value</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {Object.keys(this.state.state['quota']).map((band) => (
                                <TableRow>
                                    <TableCell>{band}</TableCell>
                                    <TableCell>
                                        <div>Total: {toReadableSize(this.state.state['quota'][band]['quota_size'])}</div>
                                        <div>Allocated: {toReadableSize(this.state.state['quota'][band]['allocated_size'])}</div>
                                        <div>Hold: {toReadableSize(this.state.state['quota'][band]['hold_size'])}</div>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </OptionalElement>
                <OptionalElement condition={this.state.state['slot']}>
                    <Title component={"h3"}>Slot</Title>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell style={{fontWeight: 'bolder'}}>Band</TableCell>
                                <TableCell style={{fontWeight: 'bolder'}}>Slots</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {Object.keys(this.state.state['slot']).map((band) => (
                                <TableRow>
                                    <TableCell>{band}</TableCell>
                                    <TableCell>{this.state.state['slot'][band].length}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </OptionalElement>
            </div>
        )
    }
}
