set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║  Building Metal 4 Image Processing Pipeline   ║"
echo "╚═══════════════════════════════════════════════╝"

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This project requires macOS with Apple Silicon"
    exit 1
fi

# Check for Metal compiler
if ! xcrun -sdk macosx metal --version 2>/dev/null; then
    echo "Error: Xcode command line tools required. Install with:"
    echo "  xcode-select --install"
    exit 1
fi

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo ">>> Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0

# Build
echo ""
echo ">>> Building..."
cmake --build . --config Release -j $(sysctl -n hw.ncpu)

echo ""
echo ">>> Build complete!"
echo ""
echo "Run benchmark:  cd build && ./m4pipe"
echo "Run tests:      cd build && ./m4pipe_test"
echo ""
echo "Python usage (if pybind11 available):"
echo "  export PYTHONPATH=$PWD:$PYTHONPATH"
echo "  python ../src/python/control_plane.py --benchmark"
