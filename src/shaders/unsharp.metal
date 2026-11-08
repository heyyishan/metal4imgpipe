#include "common.h"

// Two-pass separable Gaussian blur + unsharp mask composition
// Pass 1: Horizontal Gaussian blur
kernel void gaussian_blur_h(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant UnsharpParams& params        [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    const int R = params.kernel_radius;
    const float sigma = params.sigma;

    float4 sum = float4(0.0);
    float weight_sum = 0.0;

    for (int dx = -R; dx <= R; dx++) {
        int2 coord = int2(gid) + int2(dx, 0);
        coord.x = clamp(coord.x, 0, int(params.image_size.x) - 1);

        float w = gaussian(float(dx), sigma);
        sum += input.read(uint2(coord)) * w;
        weight_sum += w;
    }

    output.write(sum / weight_sum, gid);
}

// Pass 2: Vertical Gaussian blur
kernel void gaussian_blur_v(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant UnsharpParams& params        [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    const int R = params.kernel_radius;
    const float sigma = params.sigma;

    float4 sum = float4(0.0);
    float weight_sum = 0.0;

    for (int dy = -R; dy <= R; dy++) {
        int2 coord = int2(gid) + int2(0, dy);
        coord.y = clamp(coord.y, 0, int(params.image_size.y) - 1);

        float w = gaussian(float(dy), sigma);
        sum += input.read(uint2(coord)) * w;
        weight_sum += w;
    }

    output.write(sum / weight_sum, gid);
}

// Unsharp mask composition: output = original + amount * (original - blurred)
kernel void unsharp_compose(
    texture2d<float, access::read>  original [[texture(0)]],
    texture2d<float, access::read>  blurred  [[texture(1)]],
    texture2d<float, access::write> output   [[texture(2)]],
    constant UnsharpParams& params           [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    float4 orig = original.read(gid);
    float4 blur = blurred.read(gid);

    float4 detail = orig - blur;

    // Threshold: only sharpen where detail exceeds threshold
    float detail_mag = luminance(abs(detail.rgb));
    float mask = smoothstep(params.threshold * 0.5, params.threshold, detail_mag);

    float4 result = orig + params.amount * detail * mask;
    result = clamp(result, 0.0, 1.0);
    result.a = orig.a;

    output.write(result, gid);
}
