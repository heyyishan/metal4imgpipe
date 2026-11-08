#include <metal_stdlib>
using namespace metal;

// Simple vertex/fragment for blitting processed texture to screen
struct BlitVertexOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex BlitVertexOut blit_vertex(uint vid [[vertex_id]]) {
    // Full-screen triangle (3 vertices, no vertex buffer needed)
    BlitVertexOut out;
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    out.position = float4(positions[vid], 0.0, 1.0);
    out.texcoord = positions[vid] * 0.5 + 0.5;
    out.texcoord.y = 1.0 - out.texcoord.y; // Flip Y
    return out;
}

fragment float4 blit_fragment(
    BlitVertexOut in [[stage_in]],
    texture2d<float, access::sample> tex [[texture(0)]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return tex.sample(s, in.texcoord);
}
