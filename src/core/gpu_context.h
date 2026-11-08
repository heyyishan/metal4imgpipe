#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <IOSurface/IOSurface.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace m4pipe {

/// Encapsulates Metal device, command queues, and pipeline state management.
/// Designed for zero-contention multi-queue submission on Apple Silicon.
class GPUContext {
public:
    GPUContext();
    ~GPUContext();

    // Non-copyable
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    id<MTLDevice> device() const { return device_; }

    /// Dedicated compute queue for filter pipeline
    id<MTLCommandQueue> computeQueue() const { return compute_queue_; }

    /// Dedicated render queue for display output
    id<MTLCommandQueue> renderQueue() const { return render_queue_; }

    /// Get or create compute pipeline state for a kernel function
    id<MTLComputePipelineState> getComputePipeline(const std::string& functionName);

    /// Get render pipeline for blit-to-screen
    id<MTLRenderPipelineState> getBlitPipeline(MTLPixelFormat format);

    /// Create texture backed by IOSurface for zero-copy display path
    id<MTLTexture> createIOSurfaceTexture(uint32_t width, uint32_t height,
                                           MTLPixelFormat format = MTLPixelFormatRGBA16Float);

    /// Create standard texture for intermediate buffers
    id<MTLTexture> createTexture(uint32_t width, uint32_t height,
                                  MTLPixelFormat format = MTLPixelFormatRGBA16Float,
                                  MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

    /// Create shared buffer (CPU+GPU visible, no copy needed on unified memory)
    id<MTLBuffer> createSharedBuffer(size_t size);

    /// Device info
    std::string deviceName() const;
    size_t maxBufferLength() const;
    size_t recommendedMaxWorkingSetSize() const;

    /// Optimal threadgroup size for the device
    MTLSize optimalThreadgroupSize(id<MTLComputePipelineState> pso) const;

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> compute_queue_;
    id<MTLCommandQueue> render_queue_;
    id<MTLLibrary> library_;

    std::mutex pipeline_mutex_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> compute_pipelines_;
    id<MTLRenderPipelineState> blit_pipeline_;
    MTLPixelFormat blit_format_;
};

} // namespace m4pipe
