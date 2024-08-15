const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(express.static('public'));

app.post('/analyze', (req, res) => {
    console.log("Got tweet");
    const tweet = req.body.tweet;

    const pythonProcess = spawn('python', ['load.py']);

    pythonProcess.stdout.on('data', (data) => {
        console.log('Python script output:', data.toString());
        const result = data.toString().trim();
        res.send(result);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error from Python script: ${data}`);
        res.status(500).send('Error analyzing tweet');
    });

    pythonProcess.stdin.write(tweet);
    pythonProcess.stdin.end();
});

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

app.get('/about.html', (req, res) => {
    res.sendFile(__dirname + '/about.html');
});

app.get('/contact.html', (req, res) => {
    res.sendFile(__dirname + '/contact.html');
});

app.get('/analyze.html', (req, res) => {
    res.sendFile(__dirname + '/analyze.html');
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
