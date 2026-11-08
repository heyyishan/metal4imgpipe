#import "frame_pool.h"
#import "gpu_context.h"

namespace m4pipe {

FramePool::FramePool(GPUContext& ctx, uint32_t width, uint32_t height,
                     int intermediateCount)
    : ctx_(ctx), width_(width), height_(height), intermediate_count_(intermediateCount) {
    createResources();
}

FramePool::~FramePool() = default;

void FramePool::createResources() {
    std::lock_guard<std::mutex> lock(mutex_);

    frames_.clear();
    while (!available_.empty()) available_.pop();

    for (int i = 0; i < kMaxFramesInFlight; i++) {
        auto frame = std::make_unique<FrameResources>();

        // Input: shared storage for CPU->GPU (zero-copy on Apple Silicon)
        MTLTextureDescriptor* inputDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
            width:width_ height:height_ mipmapped:NO];
        inputDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        inputDesc.storageMode = MTLStorageModeShared;
        frame->input = [ctx_.device() newTextureWithDescriptor:inputDesc];
        frame->input.label = [NSString stringWithFormat:@"input_%d", i];

        // Output: shared for potential display path
        frame->output = [ctx_.device() newTextureWithDescriptor:inputDesc];
        frame->output.label = [NSString stringWithFormat:@"output_%d", i];

        // Intermediates: private storage (GPU-only, fastest)
        for (int j = 0; j < intermediate_count_; j++) {
            MTLTextureDescriptor* intDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                width:width_ height:height_ mipmapped:NO];
            intDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            intDesc.storageMode = MTLStorageModePrivate;
            auto tex = [ctx_.device() newTextureWithDescriptor:intDesc];
            tex.label = [NSString stringWithFormat:@"intermediate_%d_%d", i, j];
            frame->intermediates.push_back(tex);
        }

        frame->frameIndex = 0;
        frame->inFlight = false;

        available_.push(frame.get());
        frames_.push_back(std::move(frame));
    }
}

FrameResources* FramePool::acquireFrame() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !available_.empty(); });

    FrameResources* frame = available_.front();
    available_.pop();
    frame->frameIndex = frame_counter_++;
    frame->inFlight = true;
    return frame;
}

void FramePool::releaseFrame(FrameResources* frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frame->inFlight = false;
    available_.push(frame);
    cv_.notify_one();
}

void FramePool::resize(uint32_t width, uint32_t height, int intermediateCount) {
    // Wait for all frames to be available
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return available_.size() == frames_.size(); });
    }

    width_ = width;
    height_ = height;
    intermediate_count_ = intermediateCount;
    createResources();
}

} // namespace m4pipe
