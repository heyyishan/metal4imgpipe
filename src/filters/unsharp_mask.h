#pragma once

#include "filter_base.h"
#include <mutex>

namespace m4pipe {

class UnsharpMaskFilter : public FilterBase {
public:
    UnsharpMaskFilter(GPUContext& ctx);

    void prepare() override;
    void encode(id<MTLComputeCommandEncoder> encoder,
                id<MTLTexture> input,
                id<MTLTexture> output,
                FrameResources* frame) override;
    int requiredIntermediates() const override { return 2; } // Need 2 for separable blur + compose
    void updateParams(const std::vector<float>& params) override;

    void setSigma(float s);
    void setAmount(float a);
    void setThreshold(float t);

private:
    id<MTLComputePipelineState> blur_h_pso_;
    id<MTLComputePipelineState> blur_v_pso_;
    id<MTLComputePipelineState> compose_pso_;
    id<MTLBuffer> param_buffer_;
    std::mutex param_mutex_;

    float sigma_ = 2.0f;
    float amount_ = 1.5f;
    float threshold_ = 0.02f;
    int radius_ = 6;
    bool params_dirty_ = true;
};

} // namespace m4pipe
