#pragma once

#include "filter_base.h"
#include <mutex>

namespace m4pipe {

struct BilateralParams;

class BilateralFilter : public FilterBase {
public:
    BilateralFilter(GPUContext& ctx);

    void prepare() override;
    void encode(id<MTLComputeCommandEncoder> encoder,
                id<MTLTexture> input,
                id<MTLTexture> output,
                FrameResources* frame) override;
    int requiredIntermediates() const override { return 0; }
    void updateParams(const std::vector<float>& params) override;

    // Direct parameter setters
    void setSigmaSpatial(float s);
    void setSigmaRange(float r);
    void setRadius(int r);

private:
    id<MTLComputePipelineState> pso_;
    id<MTLComputePipelineState> pso_tiled_;
    id<MTLBuffer> param_buffer_;
    std::mutex param_mutex_;

    float sigma_spatial_ = 3.0f;
    float sigma_range_ = 0.1f;
    int radius_ = 3;
    bool params_dirty_ = true;
};

} // namespace m4pipe
