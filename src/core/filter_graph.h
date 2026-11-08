#pragma once

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <functional>

#import <Metal/Metal.h>

namespace m4pipe {

class FilterBase;
class GPUContext;
struct FrameResources;

/// Manages an ordered chain of filters with support for live reconfiguration.
/// Filter graph modifications are double-buffered: changes are staged and
/// swapped atomically between frames to avoid stalls.
class FilterGraph {
public:
    FilterGraph(GPUContext& ctx);
    ~FilterGraph();

    /// Add filter to end of chain
    void addFilter(std::shared_ptr<FilterBase> filter);

    /// Insert filter at position
    void insertFilter(int index, std::shared_ptr<FilterBase> filter);

    /// Remove filter by name
    void removeFilter(const std::string& name);

    /// Reorder filter chain
    void moveFilter(int fromIndex, int toIndex);

    /// Enable/disable filter by name
    void setFilterEnabled(const std::string& name, bool enabled);

    /// Get filter by name for parameter updates
    std::shared_ptr<FilterBase> getFilter(const std::string& name);

    /// Encode all active filters into a command encoder.
    /// Returns number of filters encoded.
    int encode(id<MTLComputeCommandEncoder> encoder, FrameResources* frame);

    /// Number of intermediate textures required
    int requiredIntermediates() const;

    /// List of active filter names
    std::vector<std::string> activeFilterNames() const;

    /// Prepare all filters (compile pipelines)
    void prepare();

private:
    GPUContext& ctx_;
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<FilterBase>> filters_;
};

} // namespace m4pipe
