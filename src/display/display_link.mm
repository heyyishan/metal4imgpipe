#import "display_link.h"
#import "gpu_context.h"

namespace m4pipe {

DisplayOutput::DisplayOutput(GPUContext& ctx, uint32_t width, uint32_t height)
    : ctx_(ctx) {
    // Create window
    NSRect frame = NSMakeRect(100, 100, width / 2, height / 2); // Half-size window
    window_ = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered
                      defer:NO];
    window_.title = @"Metal 4 Image Pipeline";

    // Create Metal layer
    metal_layer_ = [CAMetalLayer layer];
    metal_layer_.device = ctx_.device();
    metal_layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metal_layer_.framebufferOnly = NO;
    metal_layer_.drawableSize = CGSizeMake(width, height);
    metal_layer_.displaySyncEnabled = YES;

    // Set up view
    NSView* view = window_.contentView;
    view.wantsLayer = YES;
    view.layer = metal_layer_;

    [window_ makeKeyAndOrderFront:nil];

    // Get blit pipeline
    blit_pso_ = ctx_.getBlitPipeline(MTLPixelFormatBGRA8Unorm);
}

DisplayOutput::~DisplayOutput() {
    [window_ close];
}

void DisplayOutput::present(id<MTLTexture> texture) {
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [metal_layer_ nextDrawable];
        if (!drawable) return;

        id<MTLCommandBuffer> cmdBuf = [ctx_.renderQueue() commandBuffer];
        cmdBuf.label = @"present";

        MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = drawable.texture;
        rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> encoder = [cmdBuf renderCommandEncoderWithDescriptor:rpd];
        [encoder setRenderPipelineState:blit_pso_];
        [encoder setFragmentTexture:texture atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];

        [cmdBuf presentDrawable:drawable];
        [cmdBuf commit];
    }
}

bool DisplayOutput::isValid() const {
    return window_ != nil && window_.isVisible;
}

} // namespace m4pipe
