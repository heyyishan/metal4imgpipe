#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <functional>

#import <Metal/Metal.h>

namespace m4pipe {

class GPUContext;
class FilterGraph;
class FramePool;
class BenchmarkTracker;

/// Main pipeline engine: orchestrates frame acquisition, filter processing,
/// and output delivery at maximum throughput.
class PipelineEngine {
public:
    /// Configuration for the pipeline
    struct Config {
        uint32_t width = 3840;
        uint32_t height = 2160;
        bool enableBenchmark = true;
        int warmupFrames = 30;
        int benchmarkFrames = 300;
        bool headless = true;           // No display output
        bool generateTestPattern = true; // Generate synthetic input
    };

    PipelineEngine(const Config& config);
    ~PipelineEngine();

    /// Access filter graph for configuration
    FilterGraph& filterGraph() { return *filter_graph_; }

    /// Access GPU context
    GPUContext& gpuContext() { return *gpu_ctx_; }

    /// Run the pipeline for specified number of frames (blocking)
    void run(int numFrames);

    /// Run single frame (for integration with external loops)
    void processFrame();

    /// Run benchmark and print results
    void runBenchmark();

    /// Get benchmark tracker for external analysis
    BenchmarkTracker& benchmarkTracker() { return *benchmark_; }

    /// Stop the pipeline
    void stop();

    bool isRunning() const { return running_.load(); }

    /// Callback for frame completion
    using FrameCallback = std::function<void(id<MTLTexture> output, uint64_t frameIndex)>;
    void setFrameCallback(FrameCallback cb) { frame_callback_ = std::move(cb); }

private:
    void generateTestInput(id<MTLTexture> texture, uint64_t frameIndex);

    Config config_;
    std::unique_ptr<GPUContext> gpu_ctx_;
    std::unique_ptr<FilterGraph> filter_graph_;
    std::unique_ptr<FramePool> frame_pool_;
    std::unique_ptr<BenchmarkTracker> benchmark_;

    std::atomic<bool> running_{false};
    FrameCallback frame_callback_;

    // Timing
    CFAbsoluteTime last_frame_time_ = 0;
};

} // namespace m4pipe
