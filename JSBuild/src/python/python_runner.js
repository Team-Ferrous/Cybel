const { spawn } = require("child_process");
const path = require("path");

// Make path to your Python executable and script
const pythonExe = path.join(__dirname, "sparc_worker", "Scripts", "python.exe");
const pythonScript = path.join(__dirname, "src", "python", "sparc_server.py");

// Spawn Python process
const python = spawn(pythonExe, [pythonScript], {
  cwd: process.cwd(), // or __dirname
  stdio: ["pipe", "pipe", "pipe"], // optional but lets you handle output
});

// Capture stdout
python.stdout.on("data", (data) => {
  console.log(`[PYTHON STDOUT] ${data.toString()}`);
});

// Capture stderr
python.stderr.on("data", (data) => {
  console.error(`[PYTHON STDERR] ${data.toString()}`);
});

// Handle exit
python.on("exit", (code) => {
  console.log(`Python process exited with code ${code}`);
});

// Handle errors (like exe not found)
python.on("error", (err) => {
  console.error("Failed to start Python process:", err);
});