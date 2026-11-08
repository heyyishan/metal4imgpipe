#pragma once

#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

namespace m4pipe {

class GPUContext;
struct FrameResources;

/// Abstract base for all compute filter nodes in the pipeline.
class FilterBase {
public:
    FilterBase(const std::string& name, GPUContext& ctx)
        : name_(name), ctx_(ctx), enabled_(true) {}

    virtual ~FilterBase() = default;

    /// Encode filter commands into a compute command encoder.
    /// input/output textures are provided; filter may use intermediates from FrameResources.
    virtual void encode(id<MTLComputeCommandEncoder> encoder,
                        id<MTLTexture> input,
                        id<MTLTexture> output,
                        FrameResources* frame) = 0;

    /// Number of intermediate textures this filter requires from the frame pool
    virtual int requiredIntermediates() const { return 0; }

    /// Initialize/compile pipelines
    virtual void prepare() = 0;

    /// Update parameters (thread-safe, takes effect next frame)
    virtual void updateParams(const std::vector<float>& params) = 0;

    const std::string& name() const { return name_; }

    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
    void setEnabled(bool e) { enabled_.store(e, std::memory_order_relaxed); }

protected:
    /// Helper to dispatch with optimal threadgroup size
    void dispatchOptimal(id<MTLComputeCommandEncoder> encoder,
                         id<MTLComputePipelineState> pso,
                         uint32_t width, uint32_t height);

    std::string name_;
    GPUContext& ctx_;
    std::atomic<bool> enabled_;
};

} // namespace m4pipe
