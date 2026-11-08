#import "edge_smooth.h"
#import "gpu_context.h"
#import "frame_pool.h"
#include <simd/simd.h>

namespace m4pipe {

struct EdgeSmoothParamsCPU {
    float edge_threshold;
    float smooth_sigma;
    float detail_boost;
    int32_t kernel_radius;
    simd_uint2 image_size;
    simd_uint2 _pad0;
};

EdgeSmoothFilter::EdgeSmoothFilter(GPUContext& ctx) : FilterBase("edge_smooth", ctx) {}

void EdgeSmoothFilter::prepare() {
    pso_ = ctx_.getComputePipeline("edge_aware_smooth");
    param_buffer_ = ctx_.createSharedBuffer(sizeof(EdgeSmoothParamsCPU));
}

void EdgeSmoothFilter::encode(id<MTLComputeCommandEncoder> encoder,
                                id<MTLTexture> input,
                                id<MTLTexture> output,
                                FrameResources* frame) {
    {
        std::lock_guard<std::mutex> lock(param_mutex_);
        if (params_dirty_) {
            auto* p = static_cast<EdgeSmoothParamsCPU*>(param_buffer_.contents);
            p->edge_threshold = edge_threshold_;
            p->smooth_sigma = smooth_sigma_;
            p->detail_boost = detail_boost_;
            p->kernel_radius = radius_;
            p->image_size = simd_make_uint2((uint32_t)input.width, (uint32_t)input.height);
            params_dirty_ = false;
        }
    }

    [encoder setComputePipelineState:pso_];
    [encoder setTexture:input atIndex:0];
    [encoder setTexture:output atIndex:1];
    [encoder setBuffer:param_buffer_ offset:0 atIndex:0];
    dispatchOptimal(encoder, pso_, (uint32_t)input.width, (uint32_t)input.height);
}

void EdgeSmoothFilter::updateParams(const std::vector<float>& params) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (params.size() >= 1) edge_threshold_ = params[0];
    if (params.size() >= 2) smooth_sigma_ = params[1];
    if (params.size() >= 3) detail_boost_ = params[2];
    if (params.size() >= 4) radius_ = static_cast<int>(params[3]);
    params_dirty_ = true;
}

void EdgeSmoothFilter::setEdgeThreshold(float t) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    edge_threshold_ = t;
    params_dirty_ = true;
}

void EdgeSmoothFilter::setSmoothSigma(float s) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    smooth_sigma_ = s;
    params_dirty_ = true;
}

void EdgeSmoothFilter::setDetailBoost(float d) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    detail_boost_ = d;
    params_dirty_ = true;
}

void EdgeSmoothFilter::setRadius(int r) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    radius_ = r;
    params_dirty_ = true;
}

} // namespace m4pipe
