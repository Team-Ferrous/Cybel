// python_runner.js
const { spawn } = require("child_process");
const path = require("path");

const pythonExe = path.join(__dirname, "sparc_worker", "Scripts", "python.exe");
const pythonScript = path.join(__dirname, "src", "python", "sparc_server.py");

// Example: only call run-image or run-text
const python = spawn(pythonExe, [pythonScript, "--run-image", "target.jpg"], { cwd: __dirname });

python.stdout.on("data", data => process.stdout.write(`[PYTHON STDOUT] ${data}`));
python.stderr.on("data", data => process.stderr.write(`[PYTHON STDERR] ${data}`));
python.on("exit", code => console.log(`Python exited with code ${code}`));