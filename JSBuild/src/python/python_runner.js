// python_runner.js
const { spawn } = require("child_process");
const path = require("path");

const pythonExe = path.join(__dirname, "sparc_worker", "Scripts", "python.exe");
const pythonScript = path.join(__dirname, "src", "python", "sparc_server.py");

// -----------------------------
// Directory Checks
// -----------------------------
["agentic-file-search", "src/python", "thrml", "pygad", "kaolin", "preswald", "sprac3d_sdf", "oqtopus"].forEach(dir => {
  const fullPath = path.join(baseDir, dir);
  if (!fs.existsSync(fullPath)) {
    console.warn(`Required directory missing: ${fullPath}, some functionality may be limited.`);
  } else {
    console.log(`Directory exists: ${fullPath}`);
  }
});

// Example: only call run-image or run-text
const python = spawn(pythonExe, [pythonScript, "--run-image", "target.jpg"], { cwd: __dirname });
python.stdout.on("data", data => process.stdout.write(`[PYTHON STDOUT] ${data}`));
python.stderr.on("data", data => process.stderr.write(`[PYTHON STDERR] ${data}`));
python.on("exit", code => console.log(`Python exited with code ${code}`));