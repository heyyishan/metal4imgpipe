#import "pipeline_engine.h"
#import "gpu_context.h"
#import "filter_graph.h"
#import "frame_pool.h"
#import "benchmark.h"
#import "bilateral_filter.h"
#import "unsharp_mask.h"
#import "color_grading.h"
#import "edge_smooth.h"

#include <cmath>
#include <cstdio>

namespace m4pipe {

PipelineEngine::PipelineEngine(const Config& config) : config_(config) {
    // Initialize GPU context
    gpu_ctx_ = std::make_unique<GPUContext>();
    printf("[m4pipe] Device: %s\n", gpu_ctx_->deviceName().c_str());

    // Create filter graph
    filter_graph_ = std::make_unique<FilterGraph>(*gpu_ctx_);

    // Add default filter chain
    auto bilateral = std::make_shared<BilateralFilter>(*gpu_ctx_);
    auto unsharp = std::make_shared<UnsharpMaskFilter>(*gpu_ctx_);
    auto colorGrade = std::make_shared<ColorGradingFilter>(*gpu_ctx_);
    auto edgeSmooth = std::make_shared<EdgeSmoothFilter>(*gpu_ctx_);

    filter_graph_->addFilter(bilateral);
    filter_graph_->addFilter(colorGrade);
    filter_graph_->addFilter(unsharp);
    filter_graph_->addFilter(edgeSmooth);

    // Create frame pool with sufficient intermediates
    int intermediates = filter_graph_->requiredIntermediates();
    frame_pool_ = std::make_unique<FramePool>(*gpu_ctx_, config_.width, config_.height, intermediates);

    // Benchmark tracker
    benchmark_ = std::make_unique<BenchmarkTracker>();
    benchmark_->setDeviceName(gpu_ctx_->deviceName());
    benchmark_->setImageSize(config_.width, config_.height);
}

PipelineEngine::~PipelineEngine() {
    stop();
}

void PipelineEngine::stop() {
    running_.store(false);
}

void PipelineEngine::generateTestInput(id<MTLTexture> texture, uint64_t frameIndex) {
    // Generate a test pattern: animated gradient with detail
    uint32_t w = (uint32_t)texture.width;
    uint32_t h = (uint32_t)texture.height;
    size_t rowBytes = w * 4 * sizeof(uint16_t); // RGBA16Float

    // Use fp16 buffer
    std::vector<uint16_t> pixels(w * h * 4);

    float phase = (float)frameIndex * 0.02f;

    // Simple gradient with some high-frequency detail
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float u = (float)x / (float)w;
            float v = (float)y / (float)h;

            float r = 0.5f + 0.5f * sinf(u * 6.28f + phase);
            float g = 0.5f + 0.5f * sinf(v * 6.28f + phase * 1.3f);
            float b = 0.5f + 0.5f * sinf((u + v) * 4.0f + phase * 0.7f);

            // Add some noise/detail
            float detail = 0.1f * sinf(x * 0.1f) * cosf(y * 0.1f);
            r += detail;
            g += detail * 0.8f;
            b += detail * 0.6f;

            r = fmaxf(0.0f, fminf(1.0f, r));
            g = fmaxf(0.0f, fminf(1.0f, g));
            b = fmaxf(0.0f, fminf(1.0f, b));

            size_t idx = (y * w + x) * 4;

            // Convert float to float16 using fp16 intrinsics
            // On arm64, _Float16 is native
            _Float16 hr = (_Float16)r;
            _Float16 hg = (_Float16)g;
            _Float16 hb = (_Float16)b;
            _Float16 ha = (_Float16)1.0f;

            memcpy(&pixels[idx + 0], &hr, 2);
            memcpy(&pixels[idx + 1], &hg, 2);
            memcpy(&pixels[idx + 2], &hb, 2);
            memcpy(&pixels[idx + 3], &ha, 2);
        }
    }

    MTLRegion region = MTLRegionMake2D(0, 0, w, h);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:pixels.data()
               bytesPerRow:rowBytes];
}

void PipelineEngine::processFrame() {
    @autoreleasepool {
        // Acquire frame resources (blocks if triple buffer is full)
        FrameResources* frame = frame_pool_->acquireFrame();

        CFAbsoluteTime startTime = CFAbsoluteTimeGetCurrent();

        // Generate or load input
        if (config_.generateTestPattern) {
            generateTestInput(frame->input, frame->frameIndex);
        }

        // Create command buffer
        id<MTLCommandBuffer> cmdBuf = [gpu_ctx_->computeQueue() commandBuffer];
        cmdBuf.label = [NSString stringWithFormat:@"frame_%llu", frame->frameIndex];

        // Encode all filters
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        encoder.label = @"filter_chain";

        CFAbsoluteTime encodeStart = CFAbsoluteTimeGetCurrent();
        int filterCount = filter_graph_->encode(encoder, frame);
        [encoder endEncoding];
        CFAbsoluteTime encodeEnd = CFAbsoluteTimeGetCurrent();

        frame->cpuSubmitTime = startTime;

        // Schedule completion handler
        __block auto* framePtr = frame;
        __block auto poolPtr = frame_pool_.get();
        __block auto benchPtr = benchmark_.get();
        __block auto cbk = frame_callback_;
        __block CFAbsoluteTime submitTime = startTime;
        __block double encodeMs = (encodeEnd - encodeStart) * 1000.0;
        __block CFAbsoluteTime lastFrame = last_frame_time_;

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            CFAbsoluteTime completeTime = CFAbsoluteTimeGetCurrent();

            FrameTimingInfo info;
            info.cpuEncodeMs = encodeMs;

            // GPU timing from command buffer
            double gpuStart = cb.GPUStartTime;
            double gpuEnd = cb.GPUEndTime;
            info.gpuExecuteMs = (gpuEnd - gpuStart) * 1000.0;

            info.totalLatencyMs = (completeTime - submitTime) * 1000.0;
            info.frameIntervalMs = lastFrame > 0 ? (completeTime - lastFrame) * 1000.0 : 16.67;
            info.frameIndex = framePtr->frameIndex;

            benchPtr->recordFrame(info);

            if (cbk) {
                cbk(framePtr->output, framePtr->frameIndex);
            }

            poolPtr->releaseFrame(framePtr);
        }];

        last_frame_time_ = CFAbsoluteTimeGetCurrent();

        [cmdBuf commit];

        // Update benchmark metadata
        auto names = filter_graph_->activeFilterNames();
        benchmark_->setFilterInfo(filterCount, names);
    }
}

void PipelineEngine::run(int numFrames) {
    running_.store(true);

    for (int i = 0; i < numFrames && running_.load(); i++) {
        processFrame();
    }

    // Wait for all in-flight frames
    // Simple approach: commit a final empty command buffer and wait
    @autoreleasepool {
        id<MTLCommandBuffer> sync = [gpu_ctx_->computeQueue() commandBuffer];
        [sync commit];
        [sync waitUntilCompleted];
    }
}

void PipelineEngine::runBenchmark() {
    printf("[m4pipe] Starting benchmark: %ux%u, %d warmup, %d benchmark frames\n",
           config_.width, config_.height, config_.warmupFrames, config_.benchmarkFrames);

    benchmark_->reset();

    // Warmup
    printf("[m4pipe] Warming up (%d frames)...\n", config_.warmupFrames);
    run(config_.warmupFrames);

    // Wait for warmup to complete
    @autoreleasepool {
        id<MTLCommandBuffer> sync = [gpu_ctx_->computeQueue() commandBuffer];
        [sync commit];
        [sync waitUntilCompleted];
    }

    benchmark_->reset();

    // Benchmark
    printf("[m4pipe] Running benchmark (%d frames)...\n", config_.benchmarkFrames);
    run(config_.benchmarkFrames);

    // Wait for completion
    @autoreleasepool {
        id<MTLCommandBuffer> sync = [gpu_ctx_->computeQueue() commandBuffer];
        [sync commit];
        [sync waitUntilCompleted];
    }

    // CPU baseline
    printf("[m4pipe] Running CPU baseline...\n");
    double cpuBaseline = BenchmarkTracker::runCPUBaseline(config_.width, config_.height, 10);

    // Print results
    auto results = benchmark_->computeResults();
    BenchmarkTracker::printResults(results, cpuBaseline);
}

} // namespace m4pipe
