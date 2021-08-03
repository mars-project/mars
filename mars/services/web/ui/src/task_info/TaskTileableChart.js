import React from 'react';
import { withRouter } from "react-router-dom";
import { Grid, Paper } from '@material-ui/core';
import Title from "../Title";
import * as d3 from "d3";
import * as dagreD3 from "dagre-d3";

class tileableTileableChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedTileable: null,
            tileables: [],
            dependencies: [],
        };
    }

    fetchGraphDetail() {
        const { session_id, task_id } = this.props.match.params;

        fetch('api/session/' + session_id + '/task/' + task_id + '/tileable_graph?action=get_tileable_graph_as_json')
            .then(res => res.json())
            .then((res) => {
                this.setState({
                    tileables: res.tileables,
                    dependencies: res.dependencies,
                });
            });
    }

    componentDidMount() {
        this.g =  new dagreD3.graphlib.Graph().setGraph({});
        this.fetchGraphDetail();
    };

    componentDidUpdate(prevProps, prevStates) {
        if (prevStates.tileables !== this.state.tileables && prevStates.dependencies !== this.state.dependencies) {
            d3.select("#svg-canvas").selectAll("*").remove();

            this.g = new dagreD3.graphlib.Graph().setGraph({});
            this.state.tileables.forEach((tileable) => {
                var value = { tileable };
                value.label = tileable.tileable_name;
                value.rx = value.ry = 5;
                this.g.setNode(tileable.tileable_id, value);

                // In future fill color based on progress
                this.g.node(tileable.tileable_id).style = "fill: #9fb4c2; cursor: pointer;";
            });

            this.state.dependencies.forEach((dependency) => {
                // In future label may be named based on linkType?
                this.g.setEdge(dependency.from_tileable_id, dependency.to_tileable_id, { label: "" });
            });

            let gInstance = this.g;
            // Round the corners of the nodes
            gInstance.nodes().forEach(function (v) {
                var node = gInstance.node(v);
                node.rx = node.ry = 5;
            });

            //makes the lines smooth
            gInstance.edges().forEach(function (e) {
                var edge = gInstance.edge(e.v, e.w);
                edge.lineInterpolate = 'basis';

                edge.style = "stroke: #333";
            });

            // Create the renderer
            var render = new dagreD3.render();

            // Set up an SVG group so that we can translate the final graph.
            var svg = d3.select("#svg-canvas"),
            inner = svg.append("g");

            // Set up zoom support
            var zoom = d3.zoom().on("zoom", function (e) {
                inner.attr("transform", e.transform);
            });
            svg.call(zoom);

            if (this.state.tileables.length !== 0) {
                // Run the renderer. This is what draws the final graph.
                render(inner, this.g);
            }

            const handleClick = (e, node) => {
                const selectedTileable = this.state.tileables.filter((tileable) => tileable.tileable_id == node)[0];
                this.setState({ selectedTileable: selectedTileable});
            };

            inner.selectAll("g.node").on("click", handleClick);

            // Center the graph
            var initialScale = 0.75;
            svg.call(
                zoom.transform,
                d3.zoomIdentity.scale(initialScale)
            );

            if (this.g.graph() !== null || this.g.graph() !== undefined) {
                svg.attr("height", this.g.graph().height * initialScale + 40);
            } else {
                svg.attr("height", 40);
            }
      }
    };

    render() {
        return (
            <Grid container spacing={3} >
                 <Grid item xs={12}>
                    <Title>Graph</Title>
                </Grid>
                <Grid item xs={12} md={6}>
                    <Paper>
                        <Grid item xs={12}>
                            <svg
                                id="svg-canvas"
                                style={{ margin: 30, width: "100%", height: 700 }}
                            />
                        </Grid>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                    <Paper style={{ padding: 10 }}>
                        <Grid item xs={12}>
                            <Title>Tileable Detail: </Title>
                        </Grid>
                        <Grid item xs={12}>
                            {
                                this.state.selectedTileable
                                    ?
                                    <div>
                                        <br/>
                                        <div>Tileable ID: {this.state.selectedTileable.tileable_id}</div><br/>
                                        <div>Tileable Name: {this.state.selectedTileable.tileable_name}</div><br/>
                                    </div>
                                    :
                                    <div>
                                        Select a tileable to view its detail
                                    </div>
                            }
                        </Grid>
                    </Paper>
                </Grid>
            </Grid>
        )
    }
}

export default withRouter(tileableTileableChart);
