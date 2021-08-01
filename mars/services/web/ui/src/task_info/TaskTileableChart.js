import React from 'react';
import { withRouter } from "react-router-dom";
import * as d3 from "d3";
import * as dagreD3 from "dagre-d3";

class tileableTileableChart extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            graphModification: false,
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
        if (this.interval !== undefined)
            clearInterval(this.interval);
        this.interval = setInterval(() => this.fetchGraphDetail(), 5000);
        this.g =  new dagreD3.graphlib.Graph().setGraph({});
        this.fetchGraphDetail();
    };

    componentDidUpdate(prevProps, prevStates) {
        if (prevStates.tileables !== this.state.tileables && prevStates.dependencies !== this.state.dependencies) {
            // d3.select("#svg-canvas").selectAll("*").remove();

            this.g = new dagreD3.graphlib.Graph().setGraph({});
            this.state.tileables.forEach((tileable) => {
                var value = { tileable };
                value.label = tileable.tileable_name;
                value.rx = value.ry = 5;
                this.g.setNode(tileable.tileable_id, value);

                // In future fill color based on progress
                this.g.node(tileable.tileable_id).style = "fill: #9fb4c2";
            });

            this.state.dependencies.forEach((dependency) => {
                // In future label may be named based on linkType?
                this.g.setEdge(dependency.from_tileable_id, dependency.to_tileable_id, { label: "" });
            });

            // Create the renderer
            var render = new dagreD3.render();

            // Set up an SVG group so that we can translate the final graph.
            var svg = d3.select("svg"), inner = svg.append("g");

            // Set up zoom support
            var zoom = d3.zoom().on("zoom", function (e) {
                inner.attr("transform", e.transform);
            });
            svg.call(zoom);

            console.log(this.g, this.g.graph(), this.state.tileables);
            if (this.state.tileables.length !== 0) {
                console.log("rendered");
                // Run the renderer. This is what draws the final graph.
                render(inner, this.g);
            }

            const handleClick = (e, node) => {
                const selectedTilebale = this.state.tileables.filter((tileable) => tileable.tileable_id == node)[0];
                console.log(node, selectedTilebale);
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

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    render() {
        return (
            <div>
                <h1>Graph: </h1>
                <svg id="svg-canvas" />
            </div>
        )
    }
}

export default withRouter(tileableTileableChart);
