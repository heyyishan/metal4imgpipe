#include "common.h"

// Edge-aware smoothing: preserves edges while aggressively smoothing flat regions
// Uses structure tensor for robust edge detection

// Sobel-based gradient computation
inline float2 sobel_gradient(texture2d<float, access::read> tex, uint2 pos, uint2 size) {
    auto load = [&](int dx, int dy) -> float {
        int2 c = clamp(int2(pos) + int2(dx, dy), int2(0), int2(size) - 1);
        return luminance(tex.read(uint2(c)).rgb);
    };

    float gx = -1.0*load(-1,-1) + 1.0*load(1,-1)
               -2.0*load(-1, 0) + 2.0*load(1, 0)
               -1.0*load(-1, 1) + 1.0*load(1, 1);

    float gy = -1.0*load(-1,-1) - 2.0*load(0,-1) - 1.0*load(1,-1)
               +1.0*load(-1, 1) + 2.0*load(0, 1) + 1.0*load(1, 1);

    return float2(gx, gy);
}

kernel void edge_aware_smooth(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant EdgeSmoothParams& params     [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    const int R = params.kernel_radius;
    float4 center = input.read(gid);

    // Compute gradient magnitude at center
    float2 grad = sobel_gradient(input, gid, params.image_size);
    float grad_mag = length(grad);

    // Edge strength: 0 = flat region, 1 = strong edge
    float edge_strength = smoothstep(params.edge_threshold * 0.5,
                                      params.edge_threshold,
                                      grad_mag);

    // Adaptive sigma: smooth more in flat regions, preserve edges
    float adaptive_sigma = mix(params.smooth_sigma, params.smooth_sigma * 0.1, edge_strength);

    // If on a strong edge, optionally boost detail
    if (edge_strength > 0.5) {
        // Detail boost: mild local contrast enhancement at edges
        float3 sum = float3(0.0);
        float wsum = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 coord = clamp(int2(gid) + int2(dx, dy), int2(0), int2(params.image_size) - 1);
                float4 n = input.read(uint2(coord));
                float w = (dx == 0 && dy == 0) ? 4.0 : 1.0;
                sum += n.rgb * w;
                wsum += w;
            }
        }
        float3 local_avg = sum / wsum;
        float3 detail = center.rgb - local_avg;
        float3 result = center.rgb + detail * params.detail_boost * edge_strength;
        output.write(float4(clamp(result, 0.0, 1.0), center.a), gid);
        return;
    }

    // Weighted average for flat regions (Gaussian-weighted)
    float3 sum_color = float3(0.0);
    float sum_weight = 0.0;

    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            int2 coord = clamp(int2(gid) + int2(dx, dy), int2(0), int2(params.image_size) - 1);
            float4 neighbor = input.read(uint2(coord));

            // Spatial weight
            float w = gaussian(length(float2(dx, dy)), adaptive_sigma);

            // Range weight (mild, to avoid bleeding)
            float range_diff = length(neighbor.rgb - center.rgb);
            float wr = gaussian(range_diff, 0.1);

            float combined = w * wr;
            sum_color += neighbor.rgb * combined;
            sum_weight += combined;
        }
    }

    float3 result = sum_color / max(sum_weight, 1e-6);
    output.write(float4(result, center.a), gid);
}
