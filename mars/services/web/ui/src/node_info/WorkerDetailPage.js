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
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Tab from '@material-ui/core/Tab';
import Tabs from '@material-ui/core/Tabs';
import PropTypes from 'prop-types';
import TabPanel from './TabPanel';
import Title from '../Title';
import { useStyles } from '../Style';
import NodeEnvTab from './NodeEnvTab';
import NodeResourceTab from './NodeResourceTab';

export default function WorkerDetailPage(props) {
    const classes = useStyles();
    const [value, setValue] = React.useState(0);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    const title_text = `${props.nodeRole.replace(/\w/, (first) => first.toUpperCase())}: ${props.endpoint}`;
    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Title>{title_text}</Title>
            </Grid>
            <Grid item xs={12}>
                <Paper className={classes.paper}>
                    <Tabs value={value} onChange={handleChange}>
                        <Tab label="Environment" />
                        <Tab label="Resources" />
                    </Tabs>
                    <TabPanel value={value} index={0}>
                        <NodeEnvTab endpoint={props.endpoint} />
                    </TabPanel>
                    <TabPanel value={value} index={1}>
                        <NodeResourceTab endpoint={props.endpoint} />
                    </TabPanel>
                </Paper>
            </Grid>
        </Grid>
    );
}

WorkerDetailPage.propTypes = {
    nodeRole: PropTypes.string,
    endpoint: PropTypes.string,
};
