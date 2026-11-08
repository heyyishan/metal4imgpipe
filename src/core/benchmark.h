#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include <mutex>
#include <atomic>

namespace m4pipe {

struct FrameTimingInfo {
    double cpuEncodeMs;     // Time to encode commands on CPU
    double gpuExecuteMs;    // GPU execution time (from GPU timestamps)
    double totalLatencyMs;  // End-to-end latency
    double frameIntervalMs; // Time between frame completions
    uint64_t frameIndex;
};

struct BenchmarkResults {
    double avgFrameTimeMs;
    double p50FrameTimeMs;
    double p95FrameTimeMs;
    double p99FrameTimeMs;
    double minFrameTimeMs;
    double maxFrameTimeMs;
    double avgFPS;
    double avgLatencyMs;
    double avgGPUTimeMs;
    double avgCPUEncodeMs;
    double gpuUtilizationPct;       // Estimated
    double bandwidthGBps;           // Estimated
    int totalFrames;
    int droppedFrames;
    std::string deviceName;
    uint32_t imageWidth;
    uint32_t imageHeight;
    int filterCount;
    std::vector<std::string> filterNames;
};

class BenchmarkTracker {
public:
    BenchmarkTracker(int windowSize = 300);

    void recordFrame(const FrameTimingInfo& info);
    void reset();

    BenchmarkResults computeResults() const;

    // Real-time accessors
    double currentFPS() const;
    double currentLatencyMs() const;
    double avgFrameTimeMs() const;

    void setImageSize(uint32_t w, uint32_t h) { width_ = w; height_ = h; }
    void setDeviceName(const std::string& name) { deviceName_ = name; }
    void setFilterInfo(int count, const std::vector<std::string>& names) {
        filterCount_ = count;
        filterNames_ = names;
    }

    /// Run CPU baseline benchmark using vImage/Accelerate
    static double runCPUBaseline(uint32_t width, uint32_t height, int iterations);

    /// Print formatted results
    static void printResults(const BenchmarkResults& results, double cpuBaselineMs = 0.0);

private:
    mutable std::mutex mutex_;
    std::deque<FrameTimingInfo> history_;
    int windowSize_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::string deviceName_;
    int filterCount_ = 0;
    std::vector<std::string> filterNames_;
};

} // namespace m4pipe
