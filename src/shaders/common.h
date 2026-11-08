#ifndef METAL_COMMON_H
#define METAL_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Shared parameter structures between CPU and GPU
// All structures are aligned to 16 bytes for optimal argument buffer packing

struct BilateralParams {
    float sigma_spatial;    // Spatial domain sigma
    float sigma_range;      // Range (intensity) domain sigma
    int   kernel_radius;    // Kernel half-size
    int   _pad0;
    uint2 image_size;       // Width, Height
    uint2 _pad1;
};

struct UnsharpParams {
    float sigma;            // Gaussian blur sigma
    float amount;           // Sharpening strength (typically 0.5-3.0)
    float threshold;        // Edge threshold to avoid noise amplification
    int   kernel_radius;
    uint2 image_size;
    uint2 _pad0;
};

struct ColorGradingParams {
    float exposure;         // EV adjustment (-5 to +5)
    float contrast;         // Contrast multiplier (0.0 to 2.0, 1.0 = neutral)
    float saturation;       // Saturation multiplier (0.0 to 2.0, 1.0 = neutral)
    float temperature;      // White balance shift in Kelvin delta (-100 to +100)
    float3 lift;            // Shadow color adjustment
    float _pad0;
    float3 gamma;           // Midtone color adjustment
    float _pad1;
    float3 gain;            // Highlight color adjustment
    float _pad2;
    uint2 image_size;
    uint2 _pad3;
};

struct EdgeSmoothParams {
    float edge_threshold;   // Edge detection threshold
    float smooth_sigma;     // Smoothing strength for non-edge regions
    float detail_boost;     // Detail enhancement factor
    int   kernel_radius;
    uint2 image_size;
    uint2 _pad0;
};

// Utility functions shared across kernels
inline float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

inline float3 rgb_to_hsv(float3 c) {
    float4 K = float4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

inline float3 hsv_to_rgb(float3 c) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

inline float gaussian(float x, float sigma) {
    return exp(-0.5 * (x * x) / (sigma * sigma));
}

#endif // METAL_COMMON_H
