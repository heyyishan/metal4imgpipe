#import "unsharp_mask.h"
#import "gpu_context.h"
#import "frame_pool.h"
#include <simd/simd.h>
#include <cmath>

namespace m4pipe {

struct UnsharpParamsCPU {
    float sigma;
    float amount;
    float threshold;
    int32_t kernel_radius;
    simd_uint2 image_size;
    simd_uint2 _pad0;
};

UnsharpMaskFilter::UnsharpMaskFilter(GPUContext& ctx) : FilterBase("unsharp_mask", ctx) {}

void UnsharpMaskFilter::prepare() {
    blur_h_pso_ = ctx_.getComputePipeline("gaussian_blur_h");
    blur_v_pso_ = ctx_.getComputePipeline("gaussian_blur_v");
    compose_pso_ = ctx_.getComputePipeline("unsharp_compose");
    param_buffer_ = ctx_.createSharedBuffer(sizeof(UnsharpParamsCPU));
}

void UnsharpMaskFilter::encode(id<MTLComputeCommandEncoder> encoder,
                                id<MTLTexture> input,
                                id<MTLTexture> output,
                                FrameResources* frame) {
    {
        std::lock_guard<std::mutex> lock(param_mutex_);
        if (params_dirty_) {
            radius_ = static_cast<int>(std::ceil(sigma_ * 3.0f));
            auto* p = static_cast<UnsharpParamsCPU*>(param_buffer_.contents);
            p->sigma = sigma_;
            p->amount = amount_;
            p->threshold = threshold_;
            p->kernel_radius = radius_;
            p->image_size = simd_make_uint2((uint32_t)input.width, (uint32_t)input.height);
            params_dirty_ = false;
        }
    }

    uint32_t w = (uint32_t)input.width;
    uint32_t h = (uint32_t)input.height;

    // We need at least 2 intermediates
    id<MTLTexture> temp0 = frame->intermediates[0]; // H-blur output
    id<MTLTexture> temp1 = frame->intermediates[1]; // V-blur output (full blur)

    // Pass 1: Horizontal Gaussian blur
    [encoder setComputePipelineState:blur_h_pso_];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:temp0 atIndex:1];
    [encoder setBuffer:param_buffer_ offset:0 atIndex:0];
    dispatchOptimal(encoder, blur_h_pso_, w, h);

    // Barrier between passes
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // Pass 2: Vertical Gaussian blur
    [encoder setComputePipelineState:blur_v_pso_];
    [encoder setTexture:temp0 atIndex:0];
    [encoder setTexture:temp1 atIndex:1];
    [encoder setBuffer:param_buffer_ offset:0 atIndex:0];
    dispatchOptimal(encoder, blur_v_pso_, w, h);

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    // Pass 3: Compose (original + amount * (original - blurred))
    [encoder setComputePipelineState:compose_pso_];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:temp1 atIndex:1];
    [encoder setTexture:output atIndex:2];
    [encoder setBuffer:param_buffer_ offset:0 atIndex:0];
    dispatchOptimal(encoder, compose_pso_, w, h);
}

void UnsharpMaskFilter::updateParams(const std::vector<float>& params) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (params.size() >= 1) sigma_ = params[0];
    if (params.size() >= 2) amount_ = params[1];
    if (params.size() >= 3) threshold_ = params[2];
    params_dirty_ = true;
}

void UnsharpMaskFilter::setSigma(float s) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    sigma_ = s;
    params_dirty_ = true;
}

void UnsharpMaskFilter::setAmount(float a) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    amount_ = a;
    params_dirty_ = true;
}

void UnsharpMaskFilter::setThreshold(float t) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    threshold_ = t;
    params_dirty_ = true;
}

} // namespace m4pipe
