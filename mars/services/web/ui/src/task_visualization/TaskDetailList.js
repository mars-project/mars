import React from "react";
import {Link} from "react-router-dom";
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import Title from "../Title";
import Table from "@material-ui/core/Table";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import {useStyles} from "../Style";
import TableBody from "@material-ui/core/TableBody";

class TaskDetailList extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            tasks: [],
        };
    }

    fetchTask(sessionId) {
        fetch('api/session/' + sessionId + '/task?progress=1')
            .then(res => res.json())
            .then((res) => {
                this.setState({ tasks: res.tasks });
            });
    }

    fetchSessions() {
        fetch('api/session')
            .then(res => res.json())
            .then((res) => {
                for (let i = 0; i < res.sessions.length; i++) {
                    this.fetchTask(res.sessions[i].session_id);
                }
            })
    }

    componentDidMount() {
        if (this.interval !== undefined)
            clearInterval(this.interval);
        this.interval = setInterval(() => this.fetchSessions(), 5000);
        this.fetchSessions();
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    render() {
        return (
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell style={{fontWeight: 'bolder'}}>Task ID</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {this.state["tasks"].map((task) => (
                        <TableRow key={"task_row_" + task.task_id}>
                            <TableCell>
                                <Link to={"/tasks/" + task.session_id + "/" + task.task_id + "/graph"}>{task.task_id}</Link>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        )
    }
}

export default function TaskDetailListPage() {
    const classes = useStyles();
    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Title>Task Details</Title>
            </Grid>
            <Grid item xs={12}>
                <Paper className={classes.paper}>
                    <React.Fragment>
                        <TaskDetailList />
                    </React.Fragment>
                </Paper>
            </Grid>
        </Grid>
    )
};
