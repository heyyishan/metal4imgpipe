set -e

echo "╔═══════════════════════════════════════════════════╗"
echo "║   Metal 4 Pipeline - Full Benchmark Suite         ║"
echo "╚═══════════════════════════════════════════════════╝"

cd "$(dirname "$0")"

# Build if needed
if [[ ! -f build/m4pipe ]]; then
    echo "Building project first..."
    bash build.sh
fi

cd build

echo ""
echo "═══════════════════════════════════════════════"
echo "  1. Running Tests"
echo "═══════════════════════════════════════════════"
./m4pipe_test || echo "Some tests failed"

echo ""
echo "═══════════════════════════════════════════════"
echo "  2. 4K Benchmark (3840x2160) - All Filters"
echo "═══════════════════════════════════════════════"
./m4pipe --width 3840 --height 2160 --frames 300

echo ""
echo "═══════════════════════════════════════════════"
echo "  3. 1080p Benchmark (1920x1080) - All Filters"
echo "═══════════════════════════════════════════════"
./m4pipe --width 1920 --height 1080 --frames 500

echo ""
echo "═══════════════════════════════════════════════"
echo "  4. 4K - Bilateral Only"
echo "═══════════════════════════════════════════════"
./m4pipe --width 3840 --height 2160 --frames 300 --no-unsharp --no-colorgrade --no-edgesmooth

echo ""
echo "═══════════════════════════════════════════════"
echo "  5. 4K - Color Grading Only"
echo "═══════════════════════════════════════════════"
./m4pipe --width 3840 --height 2160 --frames 300 --no-bilateral --no-unsharp --no-edgesmooth

echo ""
echo "═══════════════════════════════════════════════"
echo "  6. 8K Stress Test (7680x4320)"
echo "═══════════════════════════════════════════════"
./m4pipe --width 7680 --height 4320 --frames 100

echo ""
echo "═══════════════════════════════════════════════"
echo "  Benchmark Complete"
echo "═══════════════════════════════════════════════"
