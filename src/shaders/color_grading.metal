#include "common.h"

// Professional color grading with lift/gamma/gain, exposure, contrast, saturation, temperature
kernel void color_grade(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant ColorGradingParams& params   [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    float4 pixel = input.read(gid);
    float3 c = pixel.rgb;

    // 1. Exposure (in EV stops)
    c *= pow(2.0, params.exposure);

    // 2. White balance / temperature shift
    // Approximate: warm shifts red up and blue down
    float temp_factor = params.temperature / 100.0;
    c.r *= (1.0 + temp_factor * 0.1);
    c.b *= (1.0 - temp_factor * 0.1);

    // 3. Lift / Gamma / Gain (CDL-style)
    // Lift affects shadows, gain affects highlights, gamma affects midtones
    float lum = luminance(c);

    // Shadow weight: stronger in dark regions
    float shadow_w = 1.0 - smoothstep(0.0, 0.5, lum);
    // Highlight weight: stronger in bright regions
    float highlight_w = smoothstep(0.5, 1.0, lum);
    // Midtone weight
    float mid_w = 1.0 - shadow_w - highlight_w;
    mid_w = max(mid_w, 0.0);

    c = c + params.lift * shadow_w;
    c = c * mix(float3(1.0), params.gain, highlight_w);
    // Gamma: power function on midtones
    float3 gamma_exp = 1.0 / max(params.gamma, float3(0.01));
    c = pow(max(c, float3(0.0)), mix(float3(1.0), gamma_exp, mid_w + 0.3));

    // 4. Contrast (pivot at 0.18 middle gray, standard for cinema)
    float pivot = 0.18;
    c = pivot + (c - pivot) * params.contrast;

    // 5. Saturation
    float mono = luminance(c);
    c = mix(float3(mono), c, params.saturation);

    // Clamp to valid range
    c = clamp(c, 0.0, 1.0);

    output.write(float4(c, pixel.a), gid);
}

// 3D LUT application kernel (for loaded LUTs)
kernel void apply_lut3d(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    texture3d<float, access::sample> lut   [[texture(2)]],
    constant ColorGradingParams& params    [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    constexpr sampler lut_sampler(filter::linear, address::clamp_to_edge);

    float4 pixel = input.read(gid);
    float3 coord = clamp(pixel.rgb, 0.0, 1.0);

    float4 graded = lut.sample(lut_sampler, coord);
    output.write(float4(graded.rgb, pixel.a), gid);
}
