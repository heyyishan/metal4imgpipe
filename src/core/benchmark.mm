#import "benchmark.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>

#import <Accelerate/Accelerate.h>

namespace m4pipe {

BenchmarkTracker::BenchmarkTracker(int windowSize) : windowSize_(windowSize) {}

void BenchmarkTracker::recordFrame(const FrameTimingInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.push_back(info);
    if ((int)history_.size() > windowSize_) {
        history_.pop_front();
    }
}

void BenchmarkTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
}

double BenchmarkTracker::currentFPS() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.size() < 2) return 0.0;
    double avgInterval = history_.back().frameIntervalMs;
    if (avgInterval <= 0) return 0.0;
    return 1000.0 / avgInterval;
}

double BenchmarkTracker::currentLatencyMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.empty()) return 0.0;
    return history_.back().totalLatencyMs;
}

double BenchmarkTracker::avgFrameTimeMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.empty()) return 0.0;
    double sum = 0.0;
    int count = std::min((int)history_.size(), 60);
    for (int i = (int)history_.size() - count; i < (int)history_.size(); i++) {
        sum += history_[i].totalLatencyMs;
    }
    return sum / count;
}

BenchmarkResults BenchmarkTracker::computeResults() const {
    std::lock_guard<std::mutex> lock(mutex_);

    BenchmarkResults r{};
    r.deviceName = deviceName_;
    r.imageWidth = width_;
    r.imageHeight = height_;
    r.filterCount = filterCount_;
    r.filterNames = filterNames_;
    r.totalFrames = (int)history_.size();

    if (history_.size() < 2) return r;

    std::vector<double> frameTimes, latencies, gpuTimes, cpuTimes, intervals;
    for (auto& h : history_) {
        frameTimes.push_back(h.totalLatencyMs);
        latencies.push_back(h.totalLatencyMs);
        gpuTimes.push_back(h.gpuExecuteMs);
        cpuTimes.push_back(h.cpuEncodeMs);
        intervals.push_back(h.frameIntervalMs);
    }

    std::sort(frameTimes.begin(), frameTimes.end());

    r.minFrameTimeMs = frameTimes.front();
    r.maxFrameTimeMs = frameTimes.back();
    r.avgFrameTimeMs = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0) / frameTimes.size();
    r.p50FrameTimeMs = frameTimes[frameTimes.size() / 2];
    r.p95FrameTimeMs = frameTimes[(int)(frameTimes.size() * 0.95)];
    r.p99FrameTimeMs = frameTimes[(int)(frameTimes.size() * 0.99)];

    double avgInterval = std::accumulate(intervals.begin(), intervals.end(), 0.0) / intervals.size();
    r.avgFPS = avgInterval > 0 ? 1000.0 / avgInterval : 0.0;

    r.avgLatencyMs = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    r.avgGPUTimeMs = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0) / gpuTimes.size();
    r.avgCPUEncodeMs = std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0) / cpuTimes.size();

    // Estimate GPU utilization
    if (avgInterval > 0) {
        r.gpuUtilizationPct = (r.avgGPUTimeMs / avgInterval) * 100.0;
    }

    // Estimate bandwidth: RGBA16Float = 8 bytes/pixel, read + write per filter pass
    double pixelsPerFrame = (double)width_ * height_;
    double bytesPerFrame = pixelsPerFrame * 8.0 * 2.0 * filterCount_; // read + write per filter
    double framesPerSec = r.avgFPS;
    r.bandwidthGBps = (bytesPerFrame * framesPerSec) / 1e9;

    // Count dropped frames (> 2x average interval)
    r.droppedFrames = 0;
    for (auto& h : history_) {
        if (h.frameIntervalMs > avgInterval * 2.0) r.droppedFrames++;
    }

    return r;
}

double BenchmarkTracker::runCPUBaseline(uint32_t width, uint32_t height, int iterations) {
    // Simulate equivalent processing using vImage / manual CPU code
    size_t pixelCount = (size_t)width * height;
    size_t bufSize = pixelCount * 4 * sizeof(float);

    std::vector<float> input(pixelCount * 4);
    std::vector<float> output(pixelCount * 4);

    // Fill with test data
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = (float)(i % 256) / 255.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        // Simulate a simple 3x3 convolution (much simpler than bilateral)
        // This represents a lower bound on CPU processing time
        vImage_Buffer src = {
            .data = input.data(),
            .height = (vImagePixelCount)height,
            .width = (vImagePixelCount)width,
            .rowBytes = width * 4 * sizeof(float)
        };
        vImage_Buffer dst = {
            .data = output.data(),
            .height = (vImagePixelCount)height,
            .width = (vImagePixelCount)width,
            .rowBytes = width * 4 * sizeof(float)
        };

        // Use vImage box filter as baseline comparison
        // This is simpler than bilateral but gives a fair comparison
        int16_t kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        vImage_Error err = vImageConvolve_PlanarF(&src, &dst, NULL, 0, 0,
                                                    (const float[]){1.0f/9, 1.0f/9, 1.0f/9,
                                                                     1.0f/9, 1.0f/9, 1.0f/9,
                                                                     1.0f/9, 1.0f/9, 1.0f/9},
                                                    3, 3, 0, kvImageEdgeExtend);
        (void)err;

        // Second pass (simulate multi-filter pipeline)
        vImageConvolve_PlanarF(&dst, &src, NULL, 0, 0,
                               (const float[]){1.0f/9, 1.0f/9, 1.0f/9,
                                                1.0f/9, 1.0f/9, 1.0f/9,
                                                1.0f/9, 1.0f/9, 1.0f/9},
                               3, 3, 0, kvImageEdgeExtend);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    return totalMs / iterations;
}

void BenchmarkTracker::printResults(const BenchmarkResults& results, double cpuBaselineMs) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║              Metal 4 Image Pipeline Benchmark Results          ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Device:     %-50s ║\n", results.deviceName.c_str());
    printf("║ Resolution: %u × %-42u ║\n", results.imageWidth, results.imageHeight);
    printf("║ Filters:    %-50d ║\n", results.filterCount);
    printf("║ Frames:     %-50d ║\n", results.totalFrames);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ FRAME TIME                                                     ║\n");
    printf("║   Average:  %8.2f ms                                        ║\n", results.avgFrameTimeMs);
    printf("║   P50:      %8.2f ms                                        ║\n", results.p50FrameTimeMs);
    printf("║   P95:      %8.2f ms                                        ║\n", results.p95FrameTimeMs);
    printf("║   P99:      %8.2f ms                                        ║\n", results.p99FrameTimeMs);
    printf("║   Min:      %8.2f ms                                        ║\n", results.minFrameTimeMs);
    printf("║   Max:      %8.2f ms                                        ║\n", results.maxFrameTimeMs);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ THROUGHPUT                                                     ║\n");
    printf("║   FPS:      %8.1f                                           ║\n", results.avgFPS);
    printf("║   Latency:  %8.2f ms                                        ║\n", results.avgLatencyMs);
    printf("║   Dropped:  %8d frames                                     ║\n", results.droppedFrames);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ GPU PERFORMANCE                                                ║\n");
    printf("║   GPU Time: %8.2f ms                                        ║\n", results.avgGPUTimeMs);
    printf("║   CPU Enc:  %8.2f ms                                        ║\n", results.avgCPUEncodeMs);
    printf("║   GPU Util: %7.1f%%                                          ║\n", results.gpuUtilizationPct);
    printf("║   BW Est:   %7.1f GB/s                                       ║\n", results.bandwidthGBps);

    if (cpuBaselineMs > 0.0) {
        double speedup = cpuBaselineMs / results.avgFrameTimeMs;
        printf("╠══════════════════════════════════════════════════════════════════╣\n");
        printf("║ CPU BASELINE COMPARISON                                        ║\n");
        printf("║   CPU Time: %8.2f ms per frame                              ║\n", cpuBaselineMs);
        printf("║   Speedup:  %7.1f× vs CPU (Accelerate/vImage)               ║\n", speedup);
    }

    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ FILTER CHAIN                                                   ║\n");
    for (auto& name : results.filterNames) {
        printf("║   → %-57s ║\n", name.c_str());
    }
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

} // namespace m4pipe
