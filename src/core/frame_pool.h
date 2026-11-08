#pragma once

#import <Metal/Metal.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

namespace m4pipe {

class GPUContext;

/// Triple-buffered frame pool for pipeline without stalls.
/// Each "frame" holds input/output textures and intermediate buffers.
struct FrameResources {
    id<MTLTexture> input;           // Source image
    id<MTLTexture> output;          // Final result
    std::vector<id<MTLTexture>> intermediates;  // Ping-pong buffers for filter chain
    id<MTLCommandBuffer> commandBuffer;
    uint64_t frameIndex;
    bool inFlight;

    // Timing
    CFAbsoluteTime cpuSubmitTime;
    CFAbsoluteTime cpuCompleteTime;
};

class FramePool {
public:
    static constexpr int kMaxFramesInFlight = 3;

    FramePool(GPUContext& ctx, uint32_t width, uint32_t height,
              int intermediateCount = 2);
    ~FramePool();

    /// Get next available frame (blocks if all frames in flight)
    FrameResources* acquireFrame();

    /// Return frame to pool after GPU completion
    void releaseFrame(FrameResources* frame);

    /// Resize all frame resources
    void resize(uint32_t width, uint32_t height, int intermediateCount);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    void createResources();

    GPUContext& ctx_;
    uint32_t width_;
    uint32_t height_;
    int intermediate_count_;

    std::vector<std::unique_ptr<FrameResources>> frames_;
    std::queue<FrameResources*> available_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<uint64_t> frame_counter_{0};
};

} // namespace m4pipe
