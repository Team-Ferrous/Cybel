#!/bin/bash
set -e

echo "⚡ Starting portable FAISS + Node setup..."

# --- 0️⃣ Ensure basic tools ---
echo "🔧 Installing build essentials..."
sudo -E apt update
sudo -E apt install -y build-essential git libopenblas-dev libomp-dev wget tar

# --- 0.5️⃣ Install portable CMake if missing ---
if ! command -v cmake &> /dev/null; then
    echo "📦 CMake not found, installing portable version..."
    CMAKE_VER=3.28.3
    CMAKE_DIR=/tmp/cmake
    mkdir -p $CMAKE_DIR
    wget -qO- https://github.com/Kitware/CMake/releases/download/v$CMAKE_VER/cmake-$CMAKE_VER-linux-x86_64.tar.gz \
        | tar xz --strip-components=1 -C $CMAKE_DIR
    export PATH=$CMAKE_DIR/bin:$PATH
    echo "✅ Portable CMake installed at $CMAKE_DIR"
fi

# --- 1️⃣ Ensure compilers are set ---
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
echo "🌿 Using compilers: $CC, $CXX"

# --- 2️⃣ Build FAISS from source ---
FAISS_DIR=/tmp/faiss
INSTALL_DIR=/usr/local

echo "📦 Cloning FAISS..."
rm -rf $FAISS_DIR
git clone https://github.com/facebookresearch/faiss.git $FAISS_DIR
cd $FAISS_DIR

echo "🔧 Configuring FAISS..."
sudo -E cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_CXX_FLAGS="-fopenmp" \
    -DCMAKE_C_FLAGS="-fopenmp"

echo "🚀 Building FAISS..."
sudo -E cmake --build   build -j$(nproc)
sudo -E cmake --install build
sudo -E ldconfig
echo "✅ FAISS installed to $INSTALL_DIR"

# --- 3️⃣ Patch Node binding.gyp ---
NODE_FAISS_DIR=~/faiss-node-native  # <-- adjust if needed
BINDING_GYP="$NODE_FAISS_DIR/binding.gyp"

if [ ! -f "$BINDING_GYP" ]; then
    echo "❌ Could not find binding.gyp at $BINDING_GYP"
    exit 1
fi

echo "✏️ Patching binding.gyp..."
cp "$BINDING_GYP" "$BINDING_GYP.bak"
sed -i '/"include_dirs": \[/a\    "/usr/local/include",' "$BINDING_GYP"
sed -i '/"libraries": \[/a\    "-L/usr/local/lib",' "$BINDING_GYP"
sed -i '/"libraries": \[/a\    "-lfaiss",' "$BINDING_GYP"
echo "✅ binding.gyp patched"

# --- 4️⃣ Set environment variables ---
export CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
echo "🌿 Environment variables set"

# --- 5️⃣ Rebuild Node module ---
echo "🔨 Rebuilding @faiss-node/native..."
cd $NODE_FAISS_DIR
rm -rf build
npm rebuild --build-from-source
echo "🎉 FAISS Node module is ready! Installing npm modules"
npm install

#if $PIP_INSTALLED then
cd ../python
pip install -r "requirements".txt
#fi

#if $GET_VERSO then
sudo apt update
sudo apt install -y \
build-essential \
libssl-dev \
zlib1g-dev \
libbz2-dev \
libreadline-dev \
libsqlite3-dev \
curl \
llvm \
libncurses5-dev \
libncursesw5-dev \
xz-utils \
tk-dev \
libffi-dev \
liblzma-dev \
git

curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Anvil
pyenv install 3.12.2
pyenv global  3.12.2
cd  ./verso/anvil/
pip install -r "requirements".txt
python scripts/install_cli_shims.py --force
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  
source ~/.bashrc 

# Build Saguaro
cd  ./Saguaro
pip install -r "requirements".txt
./build_secure.sh
#fi