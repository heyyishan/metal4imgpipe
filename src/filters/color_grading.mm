#import "color_grading.h"
#import "gpu_context.h"
#import "frame_pool.h"

namespace m4pipe {

struct ColorGradingParamsCPU {
    float exposure;
    float contrast;
    float saturation;
    float temperature;
    simd_float3 lift;
    float _pad0;
    simd_float3 gamma;
    float _pad1;
    simd_float3 gain;
    float _pad2;
    simd_uint2 image_size;
    simd_uint2 _pad3;
};

ColorGradingFilter::ColorGradingFilter(GPUContext& ctx) : FilterBase("color_grading", ctx) {}

void ColorGradingFilter::prepare() {
    pso_ = ctx_.getComputePipeline("color_grade");
    param_buffer_ = ctx_.createSharedBuffer(sizeof(ColorGradingParamsCPU));
}

void ColorGradingFilter::encode(id<MTLComputeCommandEncoder> encoder,
                                 id<MTLTexture> input,
                                 id<MTLTexture> output,
                                 FrameResources* frame) {
    {
        std::lock_guard<std::mutex> lock(param_mutex_);
        if (params_dirty_) {
            auto* p = static_cast<ColorGradingParamsCPU*>(param_buffer_.contents);
            p->exposure = exposure_;
            p->contrast = contrast_;
            p->saturation = saturation_;
            p->temperature = temperature_;
            p->lift = lift_;
            p->gamma = gamma_;
            p->gain = gain_;
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

void ColorGradingFilter::updateParams(const std::vector<float>& params) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (params.size() >= 1) exposure_ = params[0];
    if (params.size() >= 2) contrast_ = params[1];
    if (params.size() >= 3) saturation_ = params[2];
    if (params.size() >= 4) temperature_ = params[3];
    params_dirty_ = true;
}

void ColorGradingFilter::setExposure(float ev) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    exposure_ = ev;
    params_dirty_ = true;
}

void ColorGradingFilter::setContrast(float c) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    contrast_ = c;
    params_dirty_ = true;
}

void ColorGradingFilter::setSaturation(float s) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    saturation_ = s;
    params_dirty_ = true;
}

void ColorGradingFilter::setTemperature(float t) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    temperature_ = t;
    params_dirty_ = true;
}

void ColorGradingFilter::setLift(float r, float g, float b) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    lift_ = simd_make_float3(r, g, b);
    params_dirty_ = true;
}

void ColorGradingFilter::setGamma(float r, float g, float b) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    gamma_ = simd_make_float3(r, g, b);
    params_dirty_ = true;
}

void ColorGradingFilter::setGain(float r, float g, float b) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    gain_ = simd_make_float3(r, g, b);
    params_dirty_ = true;
}

} // namespace m4pipe
