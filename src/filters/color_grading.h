#pragma once

#include "filter_base.h"
#include <mutex>
#include <simd/simd.h>

namespace m4pipe {

class ColorGradingFilter : public FilterBase {
public:
    ColorGradingFilter(GPUContext& ctx);

    void prepare() override;
    void encode(id<MTLComputeCommandEncoder> encoder,
                id<MTLTexture> input,
                id<MTLTexture> output,
                FrameResources* frame) override;
    void updateParams(const std::vector<float>& params) override;

    void setExposure(float ev);
    void setContrast(float c);
    void setSaturation(float s);
    void setTemperature(float t);
    void setLift(float r, float g, float b);
    void setGamma(float r, float g, float b);
    void setGain(float r, float g, float b);

private:
    id<MTLComputePipelineState> pso_;
    id<MTLBuffer> param_buffer_;
    std::mutex param_mutex_;

    float exposure_ = 0.0f;
    float contrast_ = 1.0f;
    float saturation_ = 1.0f;
    float temperature_ = 0.0f;
    simd_float3 lift_ = {0.0f, 0.0f, 0.0f};
    simd_float3 gamma_ = {1.0f, 1.0f, 1.0f};
    simd_float3 gain_ = {1.0f, 1.0f, 1.0f};
    bool params_dirty_ = true;
};

} // namespace m4pipe
