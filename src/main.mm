#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>

#include "pipeline_engine.h"
#include "filter_graph.h"
#include "bilateral_filter.h"
#include "unsharp_mask.h"
#include "color_grading.h"
#include "edge_smooth.h"
#include "benchmark.h"

using namespace m4pipe;

void printUsage(const char* argv0) {
    printf("Usage: %s [options]\n", argv0);
    printf("\nOptions:\n");
    printf("  --width N          Image width  (default: 3840)\n");
    printf("  --height N         Image height (default: 2160)\n");
    printf("  --frames N         Number of benchmark frames (default: 300)\n");
    printf("  --warmup N         Number of warmup frames (default: 30)\n");
    printf("  --no-bilateral     Disable bilateral filter\n");
    printf("  --no-unsharp       Disable unsharp mask\n");
    printf("  --no-colorgrade    Disable color grading\n");
    printf("  --no-edgesmooth    Disable edge-aware smoothing\n");
    printf("  --help             Show this help\n");
    printf("\nExamples:\n");
    printf("  %s                           # Default 4K benchmark\n", argv0);
    printf("  %s --width 1920 --height 1080  # 1080p benchmark\n", argv0);
    printf("  %s --frames 1000               # Extended benchmark\n", argv0);
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        PipelineEngine::Config config;
        bool disableBilateral = false;
        bool disableUnsharp = false;
        bool disableColorGrade = false;
        bool disableEdgeSmooth = false;

        // Parse command line
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (arg == "--width" && i + 1 < argc) {
                config.width = std::stoul(argv[++i]);
            } else if (arg == "--height" && i + 1 < argc) {
                config.height = std::stoul(argv[++i]);
            } else if (arg == "--frames" && i + 1 < argc) {
                config.benchmarkFrames = std::stoi(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                config.warmupFrames = std::stoi(argv[++i]);
            } else if (arg == "--no-bilateral") {
                disableBilateral = true;
            } else if (arg == "--no-unsharp") {
                disableUnsharp = true;
            } else if (arg == "--no-colorgrade") {
                disableColorGrade = true;
            } else if (arg == "--no-edgesmooth") {
                disableEdgeSmooth = true;
            } else {
                printf("Unknown option: %s\n", arg.c_str());
                printUsage(argv[0]);
                return 1;
            }
        }

        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║          Metal 4 Real-Time Image Processing Pipeline       ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        try {
            PipelineEngine engine(config);

            // Configure filters based on CLI
            if (disableBilateral) engine.filterGraph().setFilterEnabled("bilateral", false);
            if (disableUnsharp)   engine.filterGraph().setFilterEnabled("unsharp_mask", false);
            if (disableColorGrade) engine.filterGraph().setFilterEnabled("color_grading", false);
            if (disableEdgeSmooth) engine.filterGraph().setFilterEnabled("edge_smooth", false);

            // Run benchmark
            engine.runBenchmark();

        } catch (const std::exception& e) {
            fprintf(stderr, "[m4pipe] Fatal error: %s\n", e.what());
            return 1;
        }

        return 0;
    }
}
