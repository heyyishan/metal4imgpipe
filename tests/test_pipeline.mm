#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cassert>
#include <cmath>

#include "pipeline_engine.h"
#include "filter_graph.h"
#include "benchmark.h"
#include "bilateral_filter.h"
#include "unsharp_mask.h"
#include "color_grading.h"
#include "edge_smooth.h"

using namespace m4pipe;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s", name); \
    try {

#define TEST_END \
        tests_passed++; \
        printf(" PASS\n"); \
    } catch (const std::exception& e) { \
        tests_failed++; \
        printf(" FAIL (%s)\n", e.what()); \
    }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond);

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) \
        throw std::runtime_error("Assertion failed: " #a " ≈ " #b);

void testGPUContext() {
    printf("\n=== GPU Context Tests ===\n");

    TEST("Create GPU context")
        GPUContext ctx;
        ASSERT(ctx.device() != nil);
        ASSERT(ctx.computeQueue() != nil);
        ASSERT(ctx.renderQueue() != nil);
        printf("[%s] ", ctx.deviceName().c_str());
    TEST_END

    TEST("Create shared buffer")
        GPUContext ctx;
        auto buf = ctx.createSharedBuffer(1024);
        ASSERT(buf != nil);
        ASSERT(buf.length >= 1024);
    TEST_END

    TEST("Create texture")
        GPUContext ctx;
        auto tex = ctx.createTexture(1920, 1080);
        ASSERT(tex != nil);
        ASSERT(tex.width == 1920);
        ASSERT(tex.height == 1080);
    TEST_END

    TEST("Get compute pipeline")
        GPUContext ctx;
        auto pso = ctx.getComputePipeline("bilateral_filter");
        ASSERT(pso != nil);
    TEST_END

    TEST("Pipeline caching")
        GPUContext ctx;
        auto pso1 = ctx.getComputePipeline("bilateral_filter");
        auto pso2 = ctx.getComputePipeline("bilateral_filter");
        ASSERT(pso1 == pso2);
    TEST_END
}

void testFramePool() {
    printf("\n=== Frame Pool Tests ===\n");

    TEST("Create frame pool")
        GPUContext ctx;
        FramePool pool(ctx, 1920, 1080, 3);
        ASSERT(pool.width() == 1920);
        ASSERT(pool.height() == 1080);
    TEST_END

    TEST("Acquire and release frame")
        GPUContext ctx;
        FramePool pool(ctx, 1920, 1080, 3);
        auto* frame = pool.acquireFrame();
        ASSERT(frame != nullptr);
        ASSERT(frame->input != nil);
        ASSERT(frame->output != nil);
        ASSERT(frame->intermediates.size() == 3);
        pool.releaseFrame(frame);
    TEST_END

    TEST("Triple buffering")
        GPUContext ctx;
        FramePool pool(ctx, 1920, 1080, 2);
        auto* f1 = pool.acquireFrame();
        auto* f2 = pool.acquireFrame();
        auto* f3 = pool.acquireFrame();
        ASSERT(f1 != f2);
        ASSERT(f2 != f3);
        ASSERT(f1 != f3);
        pool.releaseFrame(f1);
        pool.releaseFrame(f2);
        pool.releaseFrame(f3);
    TEST_END
}

void testPipelineEngine() {
    printf("\n=== Pipeline Engine Tests ===\n");

    TEST("Create engine at 1080p")
        PipelineEngine::Config cfg;
        cfg.width = 1920;
        cfg.height = 1080;
        cfg.headless = true;
        PipelineEngine engine(cfg);
    TEST_END

    TEST("Process single frame")
        PipelineEngine::Config cfg;
        cfg.width = 640;
        cfg.height = 480;
        cfg.headless = true;
        PipelineEngine engine(cfg);
        engine.processFrame();
        // Wait for completion
        @autoreleasepool {
            id<MTLCommandBuffer> sync = [engine.gpuContext().computeQueue() commandBuffer];
            [sync commit];
            [sync waitUntilCompleted];
        }
    TEST_END

    TEST("Process 10 frames")
        PipelineEngine::Config cfg;
        cfg.width = 640;
        cfg.height = 480;
        cfg.headless = true;
        PipelineEngine engine(cfg);
        engine.run(10);
    TEST_END

    TEST("Filter graph modification")
        PipelineEngine::Config cfg;
        cfg.width = 640;
        cfg.height = 480;
        cfg.headless = true;
        PipelineEngine engine(cfg);

        auto names = engine.filterGraph().activeFilterNames();
        ASSERT(names.size() == 4);

        engine.filterGraph().setFilterEnabled("bilateral", false);
        names = engine.filterGraph().activeFilterNames();
        ASSERT(names.size() == 3);

        engine.run(5);
    TEST_END
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("╔═════════════════════════════════════════╗\n");
        printf("║   Metal 4 Image Pipeline - Test Suite   ║\n");
        printf("╚═════════════════════════════════════════╝\n");

        testGPUContext();
        testFramePool();
        testPipelineEngine();

        printf("\n═══════════════════════════════════════════\n");
        printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
        printf("═══════════════════════════════════════════\n\n");

        return tests_failed > 0 ? 1 : 0;
    }
}
