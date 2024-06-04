// ./src/app.js
const express = require('express');
const bodyParser = require('body-parser');
const reportRoutes = require('./routes/reportRoutes');
const path = require("path");

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use('/api', reportRoutes);
app.use(express.static(path.join(__dirname, "../public")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../public", "index.html"));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
