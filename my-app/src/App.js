import React from 'react';
import './App.css';
import 'typeface-roboto';
import { Button, Input, FormControl, Select, MenuItem } from '@material-ui/core';
import { withStyles, lighten } from "@material-ui/core/styles";
import InputAdornment from '@material-ui/core/InputAdornment';
import SearchIcon from '@material-ui/icons/Search';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import LinearProgress from '@material-ui/core/LinearProgress';
import Typography from '@material-ui/core/Typography';
import Amplify, { API } from "aws-amplify";
import '@aws-amplify/ui/dist/style.css';
import Config from './config';


Amplify.configure({
  API: {
    endpoints: [
      {
        name: "BertEmbeddingsOpenSearch",
        endpoint: "https://aol0a4gxri.execute-api.us-east-1.amazonaws.com/development"
      }
    ]
  }
});

const styles = theme => ({
  root: {
    flexGrow: 1,
  },
  paper: {
    padding: theme.spacing(2),
    textAlign: 'center',
    height: "100%",
    color: theme.palette.text.secondary
  },
  paper2: {
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary
  },
  em: {
    backgroundColor: "#f18973"
  }
});

const BorderLinearProgress = withStyles({
  root: {
    height: 10,
    backgroundColor: lighten('#ff6c5c', 0.5),
  },
  bar: {
    borderRadius: 20,
    backgroundColor: '#ff6c5c',
  },
})(LinearProgress);

// const classes = useStyles();

class App extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      semantics: [],
      results: [],
      completed: 0,
      k: 3
    };
    this.handleSearchSubmit = this.handleSearchSubmit.bind(this);
    this.handleFormChange = this.handleFormChange.bind(this);
  }

  handleSearchSubmit(event) {
    // function for when a use submits a URL
    // if the URL bar is empty, it will remove similar photos from state
    console.log(this.state.searchText);
    if (this.state.searchText === undefined || this.state.searchText === "") {
      console.log("Empty Text field");
      this.setState({ semantics: [], completed: 0, results: [] });
    } else {
      const myInit = {
        body: { "inputs": this.state.searchText }
      };

      this.setState({ completed: 66 });

      API.post('BertEmbeddingsOpenSearch', '/', myInit)
        .then(response => {
          console.log(response);
          this.setState({
            semantics: response.body });

        })

        .catch(error => {
          console.log(error);
        });

      this.setState({ completed: 85 });

      console.log(this.state.results);
      // API.post('BertEmbeddingsOpenSearch', '/postMatch', myInit)
      //   .then(response => {
      //     // this.setState({results: []});
      //     this.setState({
      //       results: response.map(function (elem) {
      //         let result = {};
      //         result.question = elem.question;
      //         result.answer = elem.answer;
      //         return result;
      //       })
      //     });
      //     console.log(this.state.results);
      //     this.setState({ completed: 100 });
      //   })
      //   .catch(error => {
      //     console.log(error);
      //   });


    };
    event.preventDefault();
  }

  handleFormChange(event) {
    this.setState({ searchText: event.target.value });
  }


  render() {
    const { classes } = this.props;
    const createMarkup = htmlString => ({ __html: htmlString });

    return (
      <div className={classes.root}>

        <Grid container justify='center' alignItems="stretch" spacing={8} xs={12}>
          {/* <Grid item xs={10}>
            <img src={require('./images/header.jpg')} alt="Header" style={{height:"100%", width: "100%"}}/>
          </Grid> */}
          <Grid item xs={10}>
            <Typography variant="h4" style={{ textAlign: "center" }}>
              Multi-model Search
            </Typography>
          </Grid>


          <Grid item xs={10}>
            <Paper className={classes.paper}>
              <Typography variant="h6">Enter the text or upload an image </Typography>
              <p />
              <form noValidate autoComplete="off" onSubmit={this.handleSearchSubmit}>
                <Input
                  style={{ width: '80%' }}
                  placeholder="Search"
                  onChange={this.handleFormChange}
                  value={this.state.searchText}
                  id="standard-basic"
                  margin="dense"
                  width={1200}
                  // fullWidth
                  endAdornment={
                    <InputAdornment position="end">
                      <SearchIcon />
                    </InputAdornment>
                  }
                />
                <Button
                  type='submit'
                  style={{ background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)' }}
                >
                  Search
                </Button>
              </form>
            </Paper>
          </Grid>


          <Grid item xs={10}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper className={classes.paper2}>
                   
                  <Grid container spacing={2} columns={6}>
                    {this.state.semantics.map((tile) => (
                      <Grid item xs={1} sm={4} md={4}>
                        <Typography mt={2}>
                          <img style={{height:'175px',width: '175px'}} src={tile.url} />
                          <p style={{fontSize: '10px',fontStyle: 'italic', color: '#000'}}>{tile.caption}</p>
                        </Typography>
                      </Grid>
                    )
                    )
                    }
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </div>
    );
  }
}

export default withStyles(styles, { withTheme: true })(App);
