#include "common.h"

// Bilateral filter: edge-preserving smoothing
// Uses shared memory tiling for high performance on Apple Silicon
// Processes RGBA float textures

kernel void bilateral_filter(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BilateralParams& params      [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]],
    uint2 lid                             [[thread_position_in_threadgroup]],
    uint2 tgSize                          [[threads_per_threadgroup]])
{
    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    const int R = params.kernel_radius;
    const float sigma_s = params.sigma_spatial;
    const float sigma_r = params.sigma_range;

    float4 center = input.read(gid);
    float3 center_rgb = center.rgb;

    float3 sum_color = float3(0.0);
    float sum_weight = 0.0;

    // Iterate over kernel window
    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            int2 coord = int2(gid) + int2(dx, dy);

            // Clamp to image bounds
            coord = clamp(coord, int2(0), int2(params.image_size) - 1);

            float4 neighbor = input.read(uint2(coord));
            float3 neighbor_rgb = neighbor.rgb;

            // Spatial weight
            float spatial_dist = length(float2(dx, dy));
            float ws = gaussian(spatial_dist, sigma_s);

            // Range weight (intensity similarity)
            float range_dist = length(neighbor_rgb - center_rgb);
            float wr = gaussian(range_dist, sigma_r);

            float w = ws * wr;
            sum_color += neighbor_rgb * w;
            sum_weight += w;
        }
    }

    float3 result = sum_color / max(sum_weight, 1e-6);
    output.write(float4(result, center.a), gid);
}

// Optimized bilateral with threadgroup shared memory
// Uses tiled loading to reduce global memory bandwidth
kernel void bilateral_filter_tiled(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BilateralParams& params      [[buffer(0)]],
    uint2 gid                             [[thread_position_in_grid]],
    uint2 lid                             [[thread_position_in_threadgroup]],
    uint2 tgSize                          [[threads_per_threadgroup]],
    uint2 tgId                            [[threadgroup_position_in_grid]])
{
    // Tile dimensions with halo
    const int R = min(params.kernel_radius, 5); // Cap for shared memory
    const int TILE_W = 16;
    const int TILE_H = 16;
    const int SHARED_W = TILE_W + 2 * R;
    const int SHARED_H = TILE_H + 2 * R;

    threadgroup float4 shared_tile[22 * 26]; // Max (16+10) * (16+10)

    // Base position of this threadgroup tile
    int2 tg_base = int2(tgId) * int2(TILE_W, TILE_H) - int2(R);

    // Cooperative loading of shared memory
    int total_shared = SHARED_W * SHARED_H;
    int threads_in_group = TILE_W * TILE_H;
    int local_idx = int(lid.y) * TILE_W + int(lid.x);

    for (int i = local_idx; i < total_shared; i += threads_in_group) {
        int sy = i / SHARED_W;
        int sx = i % SHARED_W;
        int2 coord = tg_base + int2(sx, sy);
        coord = clamp(coord, int2(0), int2(params.image_size) - 1);
        shared_tile[i] = input.read(uint2(coord));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.x >= params.image_size.x || gid.y >= params.image_size.y) return;

    // Center pixel in shared memory
    int2 center_local = int2(lid) + int2(R);
    float4 center = shared_tile[center_local.y * SHARED_W + center_local.x];
    float3 center_rgb = center.rgb;

    float3 sum_color = float3(0.0);
    float sum_weight = 0.0;

    for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
            int2 s = center_local + int2(dx, dy);
            float4 neighbor = shared_tile[s.y * SHARED_W + s.x];
            float3 neighbor_rgb = neighbor.rgb;

            float spatial_dist = length(float2(dx, dy));
            float ws = gaussian(spatial_dist, params.sigma_spatial);

            float range_dist = length(neighbor_rgb - center_rgb);
            float wr = gaussian(range_dist, params.sigma_range);

            float w = ws * wr;
            sum_color += neighbor_rgb * w;
            sum_weight += w;
        }
    }

    float3 result = sum_color / max(sum_weight, 1e-6);
    output.write(float4(result, center.a), gid);
}
