#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>
#include <functional>
#include <atomic>

namespace m4pipe {

class GPUContext;

/// Manages a CAMetalLayer + CVDisplayLink for presenting processed frames.
/// Uses CAMetalLayer's triple buffering to avoid stalls.
class DisplayOutput {
public:
    DisplayOutput(GPUContext& ctx, uint32_t width, uint32_t height);
    ~DisplayOutput();

    /// Present a texture to screen (non-blocking, enqueues for next vsync)
    void present(id<MTLTexture> texture);

    /// Check if window is still valid
    bool isValid() const;

    CAMetalLayer* metalLayer() const { return metal_layer_; }

private:
    GPUContext& ctx_;
    NSWindow* window_;
    CAMetalLayer* metal_layer_;
    id<MTLRenderPipelineState> blit_pso_;
};

} // namespace m4pipe
