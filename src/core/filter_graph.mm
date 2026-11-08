#import "filter_graph.h"
#import "filter_base.h"
#import "gpu_context.h"
#import "frame_pool.h"

namespace m4pipe {

FilterGraph::FilterGraph(GPUContext& ctx) : ctx_(ctx) {}
FilterGraph::~FilterGraph() = default;

void FilterGraph::addFilter(std::shared_ptr<FilterBase> filter) {
    std::lock_guard<std::mutex> lock(mutex_);
    filter->prepare();
    filters_.push_back(std::move(filter));
}

void FilterGraph::insertFilter(int index, std::shared_ptr<FilterBase> filter) {
    std::lock_guard<std::mutex> lock(mutex_);
    filter->prepare();
    if (index < 0 || index > (int)filters_.size()) index = (int)filters_.size();
    filters_.insert(filters_.begin() + index, std::move(filter));
}

void FilterGraph::removeFilter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    filters_.erase(
        std::remove_if(filters_.begin(), filters_.end(),
                        [&](auto& f) { return f->name() == name; }),
        filters_.end());
}

void FilterGraph::moveFilter(int fromIndex, int toIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fromIndex < 0 || fromIndex >= (int)filters_.size()) return;
    if (toIndex < 0 || toIndex >= (int)filters_.size()) return;

    auto filter = filters_[fromIndex];
    filters_.erase(filters_.begin() + fromIndex);
    filters_.insert(filters_.begin() + toIndex, filter);
}

void FilterGraph::setFilterEnabled(const std::string& name, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& f : filters_) {
        if (f->name() == name) {
            f->setEnabled(enabled);
            break;
        }
    }
}

std::shared_ptr<FilterBase> FilterGraph::getFilter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& f : filters_) {
        if (f->name() == name) return f;
    }
    return nullptr;
}

int FilterGraph::encode(id<MTLComputeCommandEncoder> encoder, FrameResources* frame) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect active filters
    std::vector<FilterBase*> active;
    for (auto& f : filters_) {
        if (f->enabled()) active.push_back(f.get());
    }

    if (active.empty()) return 0;

    // Ping-pong between input/output and intermediates
    // For N filters:
    //   Filter 0: reads from frame->input, writes to intermediate[0] or output
    //   Filter 1: reads from intermediate[0], writes to intermediate[1] or output
    //   ...
    //   Last filter always writes to frame->output

    for (int i = 0; i < (int)active.size(); i++) {
        id<MTLTexture> input_tex;
        id<MTLTexture> output_tex;

        if (i == 0) {
            input_tex = frame->input;
        } else {
            // Read from previous filter's output
            input_tex = frame->intermediates[(i - 1) % frame->intermediates.size()];
        }

        if (i == (int)active.size() - 1) {
            output_tex = frame->output;
        } else {
            output_tex = frame->intermediates[i % frame->intermediates.size()];
            // Ensure we don't read and write to same intermediate
            if (output_tex == input_tex && frame->intermediates.size() > 1) {
                output_tex = frame->intermediates[(i + 1) % frame->intermediates.size()];
            }
        }

        active[i]->encode(encoder, input_tex, output_tex, frame);

        // Memory barrier between filter passes
        if (i < (int)active.size() - 1) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
        }
    }

    return (int)active.size();
}

int FilterGraph::requiredIntermediates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int maxReq = 2; // Minimum 2 for ping-pong

    for (auto& f : filters_) {
        maxReq = std::max(maxReq, f->requiredIntermediates() + 2);
    }

    return maxReq;
}

std::vector<std::string> FilterGraph::activeFilterNames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (auto& f : filters_) {
        if (f->enabled()) names.push_back(f->name());
    }
    return names;
}

void FilterGraph::prepare() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& f : filters_) {
        f->prepare();
    }
}

// FilterBase dispatch helper
void FilterBase::dispatchOptimal(id<MTLComputeCommandEncoder> encoder,
                                  id<MTLComputePipelineState> pso,
                                  uint32_t width, uint32_t height) {
    MTLSize gridSize = MTLSizeMake(width, height, 1);

    // Use non-uniform threadgroups for optimal coverage
    NSUInteger threadW = pso.threadExecutionWidth;
    NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger threadH = maxThreads / threadW;

    MTLSize threadgroupSize = MTLSizeMake(threadW, threadH, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

} // namespace m4pipe
