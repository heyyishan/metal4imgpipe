#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cassert>
#include <vector>
#include <cmath>

#include "gpu_context.h"
#include "frame_pool.h"
#include "bilateral_filter.h"
#include "unsharp_mask.h"
#include "color_grading.h"
#include "edge_smooth.h"

using namespace m4pipe;

// Test that each filter runs without errors and produces valid output
static void testFilter(const char* name, GPUContext& ctx, FilterBase& filter,
                       uint32_t width, uint32_t height) {
    printf("  FILTER: %-40s", name);

    @autoreleasepool {
        // Create test input
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
            width:width height:height mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        desc.storageMode = MTLStorageModeShared;

        auto input = [ctx.device() newTextureWithDescriptor:desc];
        auto output = [ctx.device() newTextureWithDescriptor:desc];

        // Fill input with test data
        size_t rowBytes = width * 8; // RGBA16Float = 8 bytes
        std::vector<uint16_t> pixels(width * height * 4);
        for (size_t i = 0; i < pixels.size(); i += 4) {
            _Float16 val = (_Float16)0.5f;
            memcpy(&pixels[i], &val, 2);
            memcpy(&pixels[i+1], &val, 2);
            memcpy(&pixels[i+2], &val, 2);
            _Float16 one = (_Float16)1.0f;
            memcpy(&pixels[i+3], &one, 2);
        }
        [input replaceRegion:MTLRegionMake2D(0, 0, width, height)
                 mipmapLevel:0 withBytes:pixels.data() bytesPerRow:rowBytes];

        // Create frame resources for intermediates
        FramePool pool(ctx, width, height, 4);
        FrameResources* frame = pool.acquireFrame();
        frame->input = input;
        frame->output = output;

        // Encode and execute
        id<MTLCommandBuffer> cmd = [ctx.computeQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        filter.encode(enc, input, output, frame);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Verify output is valid (non-zero, not NaN)
        std::vector<uint16_t> result(width * height * 4);
        [output getBytes:result.data() bytesPerRow:rowBytes
              fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];

        bool valid = true;
        int nonzeroCount = 0;
        for (size_t i = 0; i < result.size(); i += 4) {
            _Float16 r, g, b, a;
            memcpy(&r, &result[i], 2);
            memcpy(&g, &result[i+1], 2);
            memcpy(&b, &result[i+2], 2);
            memcpy(&a, &result[i+3], 2);

            if (std::isnan((float)r) || std::isnan((float)g) || std::isnan((float)b)) {
                valid = false;
                break;
            }
            if ((float)r > 0.0f || (float)g > 0.0f || (float)b > 0.0f) {
                nonzeroCount++;
            }
        }

        pool.releaseFrame(frame);

        if (valid && nonzeroCount > 0) {
            printf(" PASS\n");
        } else {
            printf(" FAIL (valid=%d, nonzero=%d)\n", valid, nonzeroCount);
        }
    }
}

// This is linked with test_pipeline.mm, so no duplicate main()
// Instead, testGPUContext() in test_pipeline.mm calls these indirectly.
// But we can add a section to the test_pipeline main:

__attribute__((constructor))
static void registerFilterTests() {
    // Filter tests are run from test_pipeline.mm's main
}

// Called from test_pipeline if needed
void runFilterTests() {
    printf("\n=== Individual Filter Tests ===\n");

    GPUContext ctx;
    const uint32_t W = 256, H = 256;

    {
        BilateralFilter f(ctx);
        f.prepare();
        testFilter("Bilateral Filter", ctx, f, W, H);
    }
    {
        UnsharpMaskFilter f(ctx);
        f.prepare();
        testFilter("Unsharp Mask", ctx, f, W, H);
    }
    {
        ColorGradingFilter f(ctx);
        f.prepare();
        testFilter("Color Grading", ctx, f, W, H);
    }
    {
        EdgeSmoothFilter f(ctx);
        f.prepare();
        testFilter("Edge-Aware Smooth", ctx, f, W, H);
    }
}
