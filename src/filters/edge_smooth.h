#pragma once

#include "filter_base.h"
#include <mutex>

namespace m4pipe {

class EdgeSmoothFilter : public FilterBase {
public:
    EdgeSmoothFilter(GPUContext& ctx);

    void prepare() override;
    void encode(id<MTLComputeCommandEncoder> encoder,
                id<MTLTexture> input,
                id<MTLTexture> output,
                FrameResources* frame) override;
    void updateParams(const std::vector<float>& params) override;

    void setEdgeThreshold(float t);
    void setSmoothSigma(float s);
    void setDetailBoost(float d);
    void setRadius(int r);

private:
    id<MTLComputePipelineState> pso_;
    id<MTLBuffer> param_buffer_;
    std::mutex param_mutex_;

    float edge_threshold_ = 0.1f;
    float smooth_sigma_ = 2.0f;
    float detail_boost_ = 1.5f;
    int radius_ = 3;
    bool params_dirty_ = true;
};

} // namespace m4pipe
