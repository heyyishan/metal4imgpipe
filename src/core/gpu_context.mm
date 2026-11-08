#import "gpu_context.h"
#include <stdexcept>

namespace m4pipe {

GPUContext::GPUContext() : blit_format_(MTLPixelFormatInvalid) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal device available");
    }

    // Create separate command queues for compute and render
    // Apple Silicon can execute compute and render concurrently on different queues
    compute_queue_ = [device_ newCommandQueue];
    compute_queue_.label = @"m4pipe.compute";

    render_queue_ = [device_ newCommandQueue];
    render_queue_.label = @"m4pipe.render";

    if (!compute_queue_ || !render_queue_) {
        throw std::runtime_error("Failed to create command queues");
    }

    // Load compiled Metal library
    NSError* error = nil;
    NSString* libPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
    if (!libPath) {
        // Try current directory (for command-line builds)
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        libPath = [cwd stringByAppendingPathComponent:@"default.metallib"];
    }
    if (!libPath || ![[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        // Try build directory
        libPath = @"default.metallib";
    }

    library_ = [device_ newLibraryWithFile:libPath error:&error];
    if (!library_) {
        // Try loading from build directory relative paths
        NSString* execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString* dir = [execPath stringByDeletingLastPathComponent];
            libPath = [dir stringByAppendingPathComponent:@"default.metallib"];
            library_ = [device_ newLibraryWithFile:libPath error:&error];
        }
    }
    if (!library_) {
        throw std::runtime_error(std::string("Failed to load Metal library: ") +
                                 (error ? error.localizedDescription.UTF8String : "unknown"));
    }
}

GPUContext::~GPUContext() {
    // ARC handles cleanup
}

id<MTLComputePipelineState> GPUContext::getComputePipeline(const std::string& functionName) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    auto it = compute_pipelines_.find(functionName);
    if (it != compute_pipelines_.end()) {
        return it->second;
    }

    NSString* name = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> func = [library_ newFunctionWithName:name];
    if (!func) {
        throw std::runtime_error("Metal function not found: " + functionName);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        throw std::runtime_error("Failed to create compute pipeline for " + functionName + ": " +
                                 (error ? error.localizedDescription.UTF8String : "unknown"));
    }

    compute_pipelines_[functionName] = pso;
    return pso;
}

id<MTLRenderPipelineState> GPUContext::getBlitPipeline(MTLPixelFormat format) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);

    if (blit_pipeline_ && blit_format_ == format) {
        return blit_pipeline_;
    }

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = [library_ newFunctionWithName:@"blit_vertex"];
    desc.fragmentFunction = [library_ newFunctionWithName:@"blit_fragment"];
    desc.colorAttachments[0].pixelFormat = format;

    NSError* error = nil;
    blit_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!blit_pipeline_) {
        throw std::runtime_error("Failed to create blit pipeline");
    }
    blit_format_ = format;
    return blit_pipeline_;
}

id<MTLTexture> GPUContext::createIOSurfaceTexture(uint32_t width, uint32_t height,
                                                    MTLPixelFormat format) {
    // Create IOSurface for zero-copy display path
    NSDictionary* props = @{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @(height),
        (id)kIOSurfaceBytesPerElement: @(8), // RGBA16Float = 8 bytes per pixel
        (id)kIOSurfacePixelFormat: @((uint32_t)'RGHA'), // Custom FourCC
    };

    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!surface) {
        throw std::runtime_error("Failed to create IOSurface");
    }

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format
        width:width
        height:height
        mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    texDesc.storageMode = MTLStorageModeShared;

    id<MTLTexture> texture = [device_ newTextureWithDescriptor:texDesc
                                                    iosurface:surface
                                                        plane:0];
    CFRelease(surface);

    if (!texture) {
        // Fallback to managed texture if IOSurface fails
        texDesc.storageMode = MTLStorageModePrivate;
        texture = [device_ newTextureWithDescriptor:texDesc];
    }

    return texture;
}

id<MTLTexture> GPUContext::createTexture(uint32_t width, uint32_t height,
                                          MTLPixelFormat format, MTLTextureUsage usage) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format
        width:width
        height:height
        mipmapped:NO];
    desc.usage = usage;
    // Private storage for GPU-only intermediates (fastest)
    desc.storageMode = MTLStorageModePrivate;

    return [device_ newTextureWithDescriptor:desc];
}

id<MTLBuffer> GPUContext::createSharedBuffer(size_t size) {
    // On Apple Silicon, shared memory is truly unified - no copy needed
    return [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
}

std::string GPUContext::deviceName() const {
    return std::string(device_.name.UTF8String);
}

size_t GPUContext::maxBufferLength() const {
    return device_.maxBufferLength;
}

size_t GPUContext::recommendedMaxWorkingSetSize() const {
    return device_.recommendedMaxWorkingSetSize;
}

MTLSize GPUContext::optimalThreadgroupSize(id<MTLComputePipelineState> pso) const {
    NSUInteger w = pso.threadExecutionWidth;          // SIMD width (typically 32)
    NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger h = maxThreads / w;
    return MTLSizeMake(w, h, 1);
}

} // namespace m4pipe
