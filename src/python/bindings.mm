#import <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pipeline_engine.h"
#include "filter_graph.h"
#include "bilateral_filter.h"
#include "unsharp_mask.h"
#include "color_grading.h"
#include "edge_smooth.h"
#include "benchmark.h"

namespace py = pybind11;
using namespace m4pipe;

PYBIND11_MODULE(m4pipe_py, m) {
    m.doc() = "Metal 4 Real-Time Image Processing Pipeline - Python Control Plane";

    // Pipeline Config
    py::class_<PipelineEngine::Config>(m, "PipelineConfig")
        .def(py::init<>())
        .def_readwrite("width", &PipelineEngine::Config::width)
        .def_readwrite("height", &PipelineEngine::Config::height)
        .def_readwrite("enable_benchmark", &PipelineEngine::Config::enableBenchmark)
        .def_readwrite("warmup_frames", &PipelineEngine::Config::warmupFrames)
        .def_readwrite("benchmark_frames", &PipelineEngine::Config::benchmarkFrames)
        .def_readwrite("headless", &PipelineEngine::Config::headless)
        .def_readwrite("generate_test_pattern", &PipelineEngine::Config::generateTestPattern);

    // Benchmark Results
    py::class_<BenchmarkResults>(m, "BenchmarkResults")
        .def_readonly("avg_frame_time_ms", &BenchmarkResults::avgFrameTimeMs)
        .def_readonly("p50_frame_time_ms", &BenchmarkResults::p50FrameTimeMs)
        .def_readonly("p95_frame_time_ms", &BenchmarkResults::p95FrameTimeMs)
        .def_readonly("p99_frame_time_ms", &BenchmarkResults::p99FrameTimeMs)
        .def_readonly("avg_fps", &BenchmarkResults::avgFPS)
        .def_readonly("avg_latency_ms", &BenchmarkResults::avgLatencyMs)
        .def_readonly("avg_gpu_time_ms", &BenchmarkResults::avgGPUTimeMs)
        .def_readonly("gpu_utilization_pct", &BenchmarkResults::gpuUtilizationPct)
        .def_readonly("bandwidth_gbps", &BenchmarkResults::bandwidthGBps)
        .def_readonly("total_frames", &BenchmarkResults::totalFrames)
        .def_readonly("dropped_frames", &BenchmarkResults::droppedFrames)
        .def_readonly("filter_names", &BenchmarkResults::filterNames);

    // Pipeline Engine
    py::class_<PipelineEngine>(m, "PipelineEngine")
        .def(py::init<const PipelineEngine::Config&>())
        .def("run_benchmark", &PipelineEngine::runBenchmark,
             "Run full benchmark with warmup")
        .def("process_frame", &PipelineEngine::processFrame,
             "Process a single frame")
        .def("run", &PipelineEngine::run,
             "Run pipeline for N frames", py::arg("num_frames"))
        .def("stop", &PipelineEngine::stop)
        .def("get_benchmark_results", [](PipelineEngine& self) {
            return self.benchmarkTracker().computeResults();
        })
        // Filter parameter access
        .def("set_bilateral_params", [](PipelineEngine& self, float sigma_s, float sigma_r, int radius) {
            auto f = std::dynamic_pointer_cast<BilateralFilter>(
                self.filterGraph().getFilter("bilateral"));
            if (f) {
                f->setSigmaSpatial(sigma_s);
                f->setSigmaRange(sigma_r);
                f->setRadius(radius);
            }
        }, py::arg("sigma_spatial") = 3.0f, py::arg("sigma_range") = 0.1f, py::arg("radius") = 3)
        .def("set_unsharp_params", [](PipelineEngine& self, float sigma, float amount, float threshold) {
            auto f = std::dynamic_pointer_cast<UnsharpMaskFilter>(
                self.filterGraph().getFilter("unsharp_mask"));
            if (f) {
                f->setSigma(sigma);
                f->setAmount(amount);
                f->setThreshold(threshold);
            }
        }, py::arg("sigma") = 2.0f, py::arg("amount") = 1.5f, py::arg("threshold") = 0.02f)
        .def("set_color_grading_params", [](PipelineEngine& self, float exposure, float contrast,
                                             float saturation, float temperature) {
            auto f = std::dynamic_pointer_cast<ColorGradingFilter>(
                self.filterGraph().getFilter("color_grading"));
            if (f) {
                f->setExposure(exposure);
                f->setContrast(contrast);
                f->setSaturation(saturation);
                f->setTemperature(temperature);
            }
        }, py::arg("exposure") = 0.0f, py::arg("contrast") = 1.0f,
           py::arg("saturation") = 1.0f, py::arg("temperature") = 0.0f)
        .def("set_edge_smooth_params", [](PipelineEngine& self, float threshold, float sigma,
                                           float detail_boost, int radius) {
            auto f = std::dynamic_pointer_cast<EdgeSmoothFilter>(
                self.filterGraph().getFilter("edge_smooth"));
            if (f) {
                f->setEdgeThreshold(threshold);
                f->setSmoothSigma(sigma);
                f->setDetailBoost(detail_boost);
                f->setRadius(radius);
            }
        }, py::arg("threshold") = 0.1f, py::arg("sigma") = 2.0f,
           py::arg("detail_boost") = 1.5f, py::arg("radius") = 3)
        .def("set_filter_enabled", [](PipelineEngine& self, const std::string& name, bool enabled) {
            self.filterGraph().setFilterEnabled(name, enabled);
        })
        .def("get_active_filters", [](PipelineEngine& self) {
            return self.filterGraph().activeFilterNames();
        })
        .def("get_device_name", [](PipelineEngine& self) {
            return self.gpuContext().deviceName();
        });

    // Standalone CPU baseline
    m.def("run_cpu_baseline", &BenchmarkTracker::runCPUBaseline,
          "Run CPU baseline benchmark",
          py::arg("width"), py::arg("height"), py::arg("iterations") = 10);
}
