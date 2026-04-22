#!/bin/bash
set -e

DEPS_DIR="${1:-/tmp/decord_deps}"

echo "=========================================="
echo "Building decord for ARM architecture"
echo "Dependencies directory: ${DEPS_DIR}"
echo "=========================================="

CMAKE_VERSION="3.19.3"

echo ">>> Checking required files..."
REQUIRED_FILES=(
    "cmake-${CMAKE_VERSION}-linux-aarch64.sh"
    "nasm-2.16.03.tar.gz"
    "yasm-1.3.0.tar.gz"
    "x264"
    "libvpx"
    "ffmpeg-4.4.2.tar.bz2"
    "decord"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "${DEPS_DIR}/${file}" ]; then
        echo "ERROR: Required file not found: ${DEPS_DIR}/${file}"
        exit 1
    fi
done
echo "All required files found."

echo ">>> Installing build tools..."
# Detect OS and use appropriate package manager
if [ -f /etc/redhat-release ] || grep -q "openEuler" /etc/os-release; then
    # CentOS/RHEL/openEuler
    yum install -y autoconf automake bzip2 bzip2-devel freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel unzip
elif grep -q "Ubuntu" /etc/os-release; then
    # Ubuntu
    apt-get update && apt-get install -y autoconf automake bzip2 libbz2-dev libfreetype-dev gcc g++ git libtool make mercurial pkg-config zlib1g-dev unzip
else
    echo "ERROR: Unsupported OS"
    exit 1
fi

echo ">>> Installing cmake ${CMAKE_VERSION}..."
cd /tmp
cp "${DEPS_DIR}/cmake-${CMAKE_VERSION}-linux-aarch64.sh" .
chmod +x "./cmake-${CMAKE_VERSION}-linux-aarch64.sh"
"./cmake-${CMAKE_VERSION}-linux-aarch64.sh" --skip-license --prefix=/usr/local
/usr/local/bin/cmake -version

mkdir -p ~/ffmpeg_sources

echo ">>> Building nasm 2.16.03..."
cd ~/ffmpeg_sources
cp "${DEPS_DIR}/nasm-2.16.03.tar.gz" .
tar xzf nasm-2.16.03.tar.gz
cd nasm-2.16.03
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
make -j$(nproc)
make install

echo ">>> Building yasm 1.3.0..."
cd ~/ffmpeg_sources
cp "${DEPS_DIR}/yasm-1.3.0.tar.gz" .
tar xzf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
make -j$(nproc)
make install

echo ">>> Building libx264..."
cd ~/ffmpeg_sources
cp -r "${DEPS_DIR}/x264" .
cd x264
export PATH="$HOME/bin:$PATH"
PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" --enable-shared --enable-pic
make -j$(nproc)
make install

echo ">>> Building libvpx..."
cd ~/ffmpeg_sources
cp -r "${DEPS_DIR}/libvpx" .
cd libvpx
export PATH="$HOME/bin:$PATH"
./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-shared --enable-pic
make -j$(nproc)
make install

echo ">>> Building ffmpeg 4.4.2..."
cd ~/ffmpeg_sources
cp "${DEPS_DIR}/ffmpeg-4.4.2.tar.bz2" .
tar xjf ffmpeg-4.4.2.tar.bz2
cd ffmpeg-4.4.2
export PATH="$HOME/bin:$PATH"
PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs=-lpthread \
  --extra-libs=-lm \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libvpx \
  --enable-libx264 \
  --enable-nonfree \
  --disable-static \
  --enable-shared \
  --enable-pic
make -j$(nproc)
make install

echo ">>> Built libraries:"
ls ~/ffmpeg_build/lib

echo ">>> Building decord..."
cd /tmp
cp -r "${DEPS_DIR}/decord" .
cd decord
mkdir -p build
cd build

# Set LD_LIBRARY_PATH to include ffmpeg_build/lib for linking
export LD_LIBRARY_PATH=~/ffmpeg_build/lib:$LD_LIBRARY_PATH

/usr/local/bin/cmake .. -DUSE_CUDA=0 -DFFMPEG_DIR=~/ffmpeg_build
make -j$(nproc)
cp libdecord.so /usr/local/lib/

# Copy all built libraries to /usr/local/lib for system-wide access
echo ">>> Copying all built libraries to /usr/local/lib..."
cp ~/ffmpeg_build/lib/*.so* /usr/local/lib/

# Update ldconfig to refresh library cache
echo ">>> Updating library cache..."
ldconfig

echo ">>> Installing Python binding..."
cd /tmp/decord/python
pip install .

echo ">>> Verifying installation..."
python -c "import decord; print(f'decord version: {decord.__version__}')"

echo ">>> Cleaning up..."
rm -rf /tmp/decord
rm -rf ~/ffmpeg_sources
rm -f "/tmp/cmake-${CMAKE_VERSION}-linux-aarch64.sh"

echo "=========================================="
echo "decord for ARM installed successfully!"
echo "=========================================="
