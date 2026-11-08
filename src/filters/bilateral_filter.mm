#import "bilateral_filter.h"
#import "gpu_context.h"
#import "frame_pool.h"

// Include the shared struct definition
#include <simd/simd.h>

namespace m4pipe {

// Mirror the Metal shader struct
struct BilateralParamsCPU {
    float sigma_spatial;
    float sigma_range;
    int32_t kernel_radius;
    int32_t _pad0;
    simd_uint2 image_size;
    simd_uint2 _pad1;
};

BilateralFilter::BilateralFilter(GPUContext& ctx) : FilterBase("bilateral", ctx) {}

void BilateralFilter::prepare() {
    pso_ = ctx_.getComputePipeline("bilateral_filter");
    pso_tiled_ = ctx_.getComputePipeline("bilateral_filter_tiled");
    param_buffer_ = ctx_.createSharedBuffer(sizeof(BilateralParamsCPU));
}

void BilateralFilter::encode(id<MTLComputeCommandEncoder> encoder,
                              id<MTLTexture> input,
                              id<MTLTexture> output,
                              FrameResources* frame) {
    // Update parameters if dirty
    {
        std::lock_guard<std::mutex> lock(param_mutex_);
        if (params_dirty_) {
            auto* p = static_cast<BilateralParamsCPU*>(param_buffer_.contents);
            p->sigma_spatial = sigma_spatial_;
            p->sigma_range = sigma_range_;
            p->kernel_radius = radius_;
            p->image_size = simd_make_uint2((uint32_t)input.width, (uint32_t)input.height);
            params_dirty_ = false;
        }
    }

    // Choose tiled version for small radii (fits in shared memory)
    auto pso = (radius_ <= 5) ? pso_tiled_ : pso_;

    [encoder setComputePipelineState:pso];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];
    [encoder setBuffer:param_buffer_ offset:0 atIndex:0];

    dispatchOptimal(encoder, pso, (uint32_t)input.width, (uint32_t)input.height);
}

void BilateralFilter::updateParams(const std::vector<float>& params) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (params.size() >= 1) sigma_spatial_ = params[0];
    if (params.size() >= 2) sigma_range_ = params[1];
    if (params.size() >= 3) radius_ = static_cast<int>(params[2]);
    params_dirty_ = true;
}

void BilateralFilter::setSigmaSpatial(float s) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    sigma_spatial_ = s;
    params_dirty_ = true;
}

void BilateralFilter::setSigmaRange(float r) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    sigma_range_ = r;
    params_dirty_ = true;
}

void BilateralFilter::setRadius(int r) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    radius_ = r;
    params_dirty_ = true;
}

} // namespace m4pipe
