const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const faissPath = path.resolve(__dirname, '../faiss-node-native');
const pythonPath = path.resolve(__dirname, '../python');

// Check folder exists
if (!fs.existsSync(faissPath)) {
  console.error('faiss-node-native folder not found:', faissPath);
  process.exit(1);
}

console.log('🛠️  Building FAISS native module...');

// Try node-gyp first (Windows / Electron)
try {
  execSync('node-gyp rebuild', { cwd: faissPath, stdio: 'inherit' });
} catch (e) {
  console.warn('⚠️  node-gyp failed, trying CMake fallback...');
  try {
    execSync('cmake --build build --config Release', { cwd: faissPath, stdio: 'inherit' });
  } catch (e2) {
    console.error('❌  FAISS build failed.');
    process.exit(1);
  }
}

console.log('✅  FAISS build completed successfully.');

// Try Python setup
try {
  execSync('pip install -r requirements.txt', { cwd: pythonPath, stdio: 'inherit' });
} catch (e) {
  console.error('❌  PythonLibs build failed.');
}
console.log('✅  Python Libs installed to repo successfully.');