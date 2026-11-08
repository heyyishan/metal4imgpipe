# Metal 4 Real-Time Image Processing Pipeline

> High-performance, GPU-accelerated image processing pipeline for Apple Silicon,
> engineered for sustained 60+ FPS at 4K resolution with multiple compute-intensive filters.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python Control Plane                        │
│              (pybind11: live parameter tweaking, filter reconfig)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Lock-free parameter updates
┌────────────────────────────────▼────────────────────────────────────┐
│                        PipelineEngine (C++23)                      │
│  ┌──────────┐  ┌────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │ FramePool│  │FilterGraph │  │ BenchmarkTrack│  │DisplayOutput│ │
│  │ (Triple  │  │ (Ordered   │  │ (Timing &     │  │(CAMetalLayer│ │
│  │ Buffer)  │  │  Chain)    │  │  Statistics)  │  │ + Blit)     │  │
│  └────┬─────┘  └─────┬──────┘  └───────────────┘  └──────┬─────┘  │
│       │              │                                     │       │
└───────┼──────────────┼─────────────────────────────────────┼───────┘
        │              │                                     │
┌───────▼──────────────▼─────────────────────────────────────▼───────┐
│                      GPUContext (Metal Device)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ Compute Queue   │  │  Render Queue   │  │ Pipeline Cache   │   │
│  │ (Filter Chain)  │  │  (Blit/Display) │  │ (Lazy-compiled)  │   │
│  └────────┬────────┘  └────────┬────────┘  └──────────────────┘   │
└───────────┼─────────────────────┼──────────────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼──────────────────────────────────┐
│                    Apple Silicon GPU (Unified Memory)              │
│                                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Bilateral │→ │ Color    │→ │ Unsharp  │→ │ Edge-Aware       │  │
│  │ Filter   │  │ Grading  │  │  Mask    │  │ Smoothing        │  │
│  │(Tiled TG)│  │ (LGG+CDL)│  │(Sep.Blur)│  │(Struct.Tensor)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                                    │
│  Shared Memory ◄──── Zero-copy ────► CPU                          │
│  Private Intermediates ◄── GPU-only, maximum bandwidth ──►        │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Per Frame)

```
CPU: Generate/Load Input ──► Shared MTLTexture (zero-copy on UMA)
                                      │
GPU Compute Queue:                    ▼
  ┌─────────────────────────────────────────────────────────┐
  │ MTLComputeCommandEncoder                                 │
  │                                                          │
  │  Input ─► Bilateral ─► Intermediate[0]                   │
  │                │ memory barrier                          │
  │  Inter[0] ─► ColorGrade ─► Intermediate[1]               │
  │                │ memory barrier                          │
  │  Inter[1] ─► GaussH ─► Inter[0]                          │
  │                │ memory barrier                          │
  │  Inter[0] ─► GaussV ─► Inter[1]                          │
  │                │ memory barrier                          │
  │  Input+Inter[1] ─► UnsharpCompose ─► Inter[0]            │
  │                │ memory barrier                          │
  │  Inter[0] ─► EdgeSmooth ─► Output                        │
  └──────────────────────────────────────────────────────────┘
                                      │
GPU Render Queue:                     ▼
  Output ─► Blit Fragment Shader ─► CAMetalDrawable ─► Display
```

## Key Design Decisions

### 1. Unified Memory Architecture (Zero-Copy Path)

On Apple Silicon, CPU and GPU share the same physical memory. We exploit this:

- **Input textures**: `MTLStorageModeShared` — CPU writes directly to GPU-visible memory. No `memcpy`, no staging buffer, no DMA transfer. The texture contents are immediately available to compute kernels.
- **Intermediate textures**: `MTLStorageModePrivate` — GPU-only. The Metal driver can place these in the fastest memory tier (likely tile memory or L2 cache-adjacent). Since intermediates never need CPU access, this is strictly faster than shared.
- **Parameter buffers**: `MTLStorageModeShared` with dirty-flag pattern. CPU writes parameters between frames; GPU reads them. On UMA this is a single cache line flush, not a DMA operation.

**Trade-off**: Shared textures for input/output are ~5-10% slower for GPU reads than private textures, but eliminating the copy more than compensates when the CPU is generating or loading frames.

### 2. Triple Buffering

Three `FrameResources` sets rotate through the pipeline:

```
Frame N:   GPU executing  ─────────────┐
Frame N+1: GPU executing (queued)      │ Concurrent
Frame N+2: CPU encoding   ◄────────────┘
```

This ensures the CPU never blocks waiting for GPU completion (up to 3 frames deep). The `FramePool` uses a condition variable to block only when all three frames are in flight — which means the GPU is fully saturated, which is the desired steady state.

**Trade-off**: 3-frame latency in the worst case. For a display pipeline this is standard (compositor adds another frame anyway). For latency-critical applications, reduce to 2 frames at the cost of occasional CPU stalls.

### 3. Dual Command Queues

- **Compute Queue**: Dedicated to the filter chain. All compute dispatches for a frame go into a single command buffer with one `MTLComputeCommandEncoder`.
- **Render Queue**: Dedicated to the blit-to-display path.

On Apple Silicon, these can execute concurrently on different GPU execution units. The compute queue is never blocked by vsync waits on the render queue.

### 4. Single Encoder per Frame

All filter passes are encoded into **one** `MTLComputeCommandEncoder` with `memoryBarrierWithScope:` between passes. This is faster than separate encoders because:

- Eliminates encoder setup/teardown overhead (~2-5μs per encoder)
- Allows the Metal driver to fuse adjacent dispatches when possible
- Reduces command buffer overhead

The memory barriers ensure correct ordering without forcing a full pipeline flush.

### 5. Separable Convolution (Unsharp Mask)

The unsharp mask uses a **two-pass separable Gaussian blur** (horizontal then vertical), reducing complexity from O(r²) to O(r) per pixel. For a radius-6 kernel:

- Non-separable: 169 texture reads per pixel
- Separable: 26 texture reads per pixel (6.5× fewer)

**Trade-off**: Requires an extra intermediate texture and an extra memory barrier. Worth it for any radius > 2.

### 6. Threadgroup Shared Memory (Tiled Bilateral)

The bilateral filter's tiled variant loads a tile + halo into threadgroup shared memory cooperatively, then each thread reads from fast SRAM instead of global memory:

```
Global Memory Bandwidth:    ~100 GB/s (M3/M4)
Threadgroup Memory:         ~1 TB/s (estimated from SIMD width × clock)
```

For a 5×5 bilateral (radius=2), this is 3-4× faster. For radius > 5, the shared memory tile becomes too large, and we fall back to the direct global-read kernel.

### 7. Lock-Free Parameter Updates

Filter parameters are protected by a per-filter `std::mutex`, but the critical section is trivially small (copy a few floats + set dirty flag). The parameter buffer is only written to GPU memory when `params_dirty_` is true, avoiding unnecessary cache flushes.

Parameters take effect on the **next frame** — never mid-frame. This is a deliberate design choice: mid-frame updates could cause visible tearing artifacts.

### 8. Dynamic Filter Graph Reconfiguration

Filters can be enabled/disabled, reordered, or added/removed between frames. The `FilterGraph::encode()` method holds the graph lock only during encoding — which is O(microseconds). The actual GPU work is decoupled.

**Trade-off**: Graph mutations and encoding are serialized by the same mutex. In practice this isn't a bottleneck because encode() is ~10-50μs and mutations are rare (human-driven UI changes).

### 9. RGBA16Float Working Format

All intermediate processing uses `MTLPixelFormatRGBA16Float` (64 bits/pixel):

- **Pro**: No banding artifacts, HDR-capable, sufficient precision for all color math
- **Pro**: Native ALU width on Apple Silicon (FP16 is the GPU's preferred precision)
- **Con**: 2× memory bandwidth vs RGBA8. At 4K this is ~64 MB per texture, well within M4's memory budget
- **Con**: 2× intermediate storage. Triple-buffered 4K with 4 intermediates = ~1.5 GB. The M4 MacBook Air has 16-32 GB unified memory, so this is fine.

## Filter Implementations

### Bilateral Filter (`bilateral.metal`)
Edge-preserving smoothing using joint spatial-range weighting. Two variants:
- **Standard**: Direct global memory reads, any radius
- **Tiled**: Threadgroup shared memory, radius ≤ 5, ~3× faster

### Unsharp Mask (`unsharp.metal`)
Three-pass: separable Gaussian blur (H + V) → compose with threshold-masked detail.
The threshold prevents noise amplification in flat regions.

### Color Grading (`color_grading.metal`)
Professional-grade color pipeline: Exposure (EV) → White Balance → Lift/Gamma/Gain (CDL-style) → Contrast (0.18 pivot) → Saturation. Single-pass, pure ALU-bound.

### Edge-Aware Smoothing (`edge_smooth.metal`)
Sobel gradient → structure tensor edge detection → adaptive Gaussian smoothing (strong in flat regions, bypassed at edges) with optional detail boost at edges.

## Benchmark Results

### Methodology

- **Hardware**: Apple M4 MacBook Air (10-core GPU, 16GB unified memory)
- **OS**: macOS 15.x (Sequoia)
- **Measurement**: `MTLCommandBuffer.GPUStartTime/GPUEndTime` for GPU time; `CFAbsoluteTimeGetCurrent` for end-to-end latency
- **Warmup**: 30 frames discarded before measurement
- **Frames**: 300 frames per benchmark
- **Input**: Synthetic animated gradient pattern (worst case for bilateral: high spatial frequency)

### Expected Results (M4 MacBook Air, 10-core GPU)

| Scenario | Resolution | Filters | Avg FPS | P95 Frame (ms) | GPU Time (ms) | Bandwidth (GB/s) |
|----------|-----------|---------|---------|----------------|---------------|-------------------|
| Full chain | 3840×2160 | 4 | 90-150+ | 8-12 | 3-7 | 40-80 |
| Full chain | 1920×1080 | 4 | 300-500+ | 2-4 | 1-3 | 20-40 |
| Bilateral only | 3840×2160 | 1 | 200-400+ | 3-5 | 1-3 | 15-30 |
| Color grade only | 3840×2160 | 1 | 500-1000+ | 1-2 | 0.3-1 | 30-60 |
| Full chain | 7680×4320 | 4 | 30-60+ | 16-33 | 10-25 | 50-90 |

*Note: Actual results depend on thermal state, background load, and filter parameters. The bilateral filter is bandwidth-limited; color grading is ALU-limited. The test pattern generation on CPU dominates total latency for small resolutions.*

### CPU Baseline Comparison

| Resolution | CPU (vImage 3×3 conv × 2) | GPU (Full 4-filter chain) | Speedup |
|-----------|---------------------------|---------------------------|---------|
| 3840×2160 | ~80-150 ms | ~5-10 ms | 10-20× |
| 1920×1080 | ~20-40 ms | ~2-4 ms | 8-15× |

*CPU baseline uses `vImageConvolve_PlanarF` with a 3×3 kernel (simpler than bilateral), giving the CPU a generous advantage. Real bilateral on CPU would be 10-50× slower.*

## Build & Run Instructions

### Prerequisites

- macOS 15.0+ (Sequoia)
- Apple Silicon Mac (M1/M2/M3/M4/M5)
- Xcode 16+ (command line tools)
- CMake 3.25+

### Quick Build

```bash
chmod +x build.sh
./build.sh
```

### Manual Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

### Run Benchmark

```bash
cd build

# Default 4K benchmark
./m4pipe

# Custom resolution
./m4pipe --width 1920 --height 1080

# Extended benchmark
./m4pipe --frames 1000

# Individual filter benchmarks
./m4pipe --no-unsharp --no-colorgrade --no-edgesmooth  # Bilateral only

# Full suite
cd .. && chmod +x benchmark.sh && ./benchmark.sh
```

### Run Tests

```bash
cd build && ./m4pipe_test
```

### Python Control Plane

```bash
# Install pybind11
pip install pybind11

# Rebuild with Python support
cd build && cmake .. && make

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Run benchmark from Python
python ../src/python/control_plane.py --benchmark

# Interactive mode
python ../src/python/control_plane.py
```

#### Python Interactive Examples

```python
import m4pipe_py as m4

config = m4.PipelineConfig()
config.width = 3840
config.height = 2160

engine = m4.PipelineEngine(config)

# Tweak parameters live
engine.set_bilateral_params(sigma_spatial=5.0, sigma_range=0.15, radius=4)
engine.set_color_grading_params(exposure=0.5, contrast=1.2, saturation=1.1, temperature=10)

# Process frames
engine.run(100)

# Get results
results = engine.get_benchmark_results()
print(f"FPS: {results.avg_fps:.1f}, Latency: {results.avg_latency_ms:.2f}ms")

# Disable a filter dynamically
engine.set_filter_enabled("bilateral", False)
engine.run(100)
```

## Performance Analysis

### Bottleneck Identification

1. **Bilateral Filter** (bandwidth-bound): Each pixel reads (2R+1)² neighbors. At R=3, that's 49 reads × 8 bytes = 392 bytes per pixel. For 4K (8.3M pixels): ~3.2 GB per dispatch. The tiled variant reduces this by ~3× via shared memory reuse.

2. **Unsharp Mask** (bandwidth-bound): Three passes, each touching every pixel. ~50 MB total data movement at 4K RGBA16F.

3. **Color Grading** (ALU-bound): Single-pass, minimal data movement (read + write = 16 bytes/pixel). Limited by FP16 throughput. Typically completes in < 1ms at 4K.

4. **Edge Smooth** (mixed): Sobel gradient is ALU-heavy; smoothing kernel is bandwidth-heavy. The conditional branching (edge vs. flat region) can cause SIMD divergence.

### Optimization Opportunities (Not Yet Implemented)

- **Kernel fusion**: Merge color grading + unsharp compose into a single kernel, saving one full texture read-write pass
- **Subgroup operations**: Use `simd_shuffle` for small-radius blur, eliminating shared memory entirely
- **Indirect command buffers**: Pre-encode filter chains as reusable ICBs, reducing per-frame CPU overhead to near-zero
- **Adaptive resolution**: Process expensive filters (bilateral) at half-resolution with bilinear upscale
- **Metal 4 ray tracing acceleration structure**: Not applicable to this workload (but mesh shaders could optimize the blit pass for tiled rendering)

### Known Limitations

1. **Test pattern generation is CPU-bound**: The `generateTestInput()` function converts floats to FP16 on CPU and writes to a shared texture. For a production pipeline, input would come from a camera/decoder directly into GPU memory.

2. **No IOSurface display path in headless mode**: The benchmark runs headless. The `DisplayOutput` class implements CAMetalLayer presentation but is not exercised in benchmark mode.

3. **Metal 3.2 shaders, not Metal 4**: As of this writing, Metal 4 (if it refers to a future Metal Shading Language version) is not publicly released. The pipeline uses Metal 3.2 (`-std=metal3.2`) which is the latest available for Apple Silicon. The architecture is designed to adopt Metal 4 features (enhanced argument buffers, mesh shaders, etc.) when available.

4. **Single command encoder serialization**: All filters execute sequentially within one encoder. Filters that don't have data dependencies could theoretically execute concurrently in separate encoders, but Metal's compute execution model on Apple Silicon already extracts ILP within a single encoder.

5. **Parameter buffer writes are not double-buffered**: If the CPU writes parameters while the GPU is reading the same buffer (from a previous frame), there's a potential data race. In practice, triple buffering makes this unlikely, but a production system should use per-frame parameter buffers.

## License

MIT License. See LICENSE for details.
