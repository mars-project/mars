import React from 'react';
import PropTypes from 'prop-types';
import { Grid, Paper } from '@material-ui/core';
import Title from '../Title';

class TileableDetail extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    render() {
        if (this.props === undefined) {
            return <div></div>;
        }

        return (
            <Grid item xs={12} md={6}>
                <Paper style={{ padding: 10 }}>
                    <Grid item xs={12}>
                        <Title>Tileable Detail: </Title>
                    </Grid>
                    <Grid item xs={12}>
                        {
                            this.props.selectedTileable
                                ?
                                <div>
                                    <br/>
                                    <div>Tileable ID: <br/>{this.props.selectedTileable.tileable_id}</div><br/>
                                    <div>Tileable Name: <br/>{this.props.selectedTileable.tileable_name}</div><br/>
                                </div>
                                :
                                <div>
                                    Select a tileable to view its detail
                                </div>
                        }
                    </Grid>
                </Paper>
            </Grid>
        );
    }
}

TileableDetail.propTypes = {
    selectedTileable: PropTypes.shape({
        tileable_id: PropTypes.string,
        tileable_name: PropTypes.string,
    }),
};

export default TileableDetail;
