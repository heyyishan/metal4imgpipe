"""
Metal 4 Image Pipeline - Python Control Plane

Interactive control of the real-time image processing pipeline.
Supports live parameter tweaking, filter graph modification, and benchmarking.

Usage:
    python control_plane.py              # Interactive mode
    python control_plane.py --benchmark  # Run benchmark and exit
"""

import sys
import time
import threading

try:
    import m4pipe_py as m4
except ImportError:
    print("Error: m4pipe_py module not found.")
    print("Build with: mkdir build && cd build && cmake .. && make")
    print("Then: export PYTHONPATH=$PWD/build:$PYTHONPATH")
    sys.exit(1)


class PipelineController:
    """High-level controller for the Metal 4 image pipeline."""

    def __init__(self, width=3840, height=2160):
        config = m4.PipelineConfig()
        config.width = width
        config.height = height
        config.headless = True
        config.generate_test_pattern = True
        config.warmup_frames = 30
        config.benchmark_frames = 300

        self.engine = m4.PipelineEngine(config)
        self.width = width
        self.height = height
        self._running = False

    def run_benchmark(self):
        """Run full benchmark suite."""
        print(f"\n{'='*60}")
        print(f"Running benchmark at {self.width}x{self.height}")
        print(f"{'='*60}")
        self.engine.run_benchmark()
        return self.engine.get_benchmark_results()

    def set_bilateral(self, sigma_spatial=3.0, sigma_range=0.1, radius=3):
        """Update bilateral filter parameters."""
        self.engine.set_bilateral_params(sigma_spatial, sigma_range, radius)

    def set_unsharp(self, sigma=2.0, amount=1.5, threshold=0.02):
        """Update unsharp mask parameters."""
        self.engine.set_unsharp_params(sigma, amount, threshold)

    def set_color_grading(self, exposure=0.0, contrast=1.0, saturation=1.0, temperature=0.0):
        """Update color grading parameters."""
        self.engine.set_color_grading_params(exposure, contrast, saturation, temperature)

    def set_edge_smooth(self, threshold=0.1, sigma=2.0, detail_boost=1.5, radius=3):
        """Update edge-aware smoothing parameters."""
        self.engine.set_edge_smooth_params(threshold, sigma, detail_boost, radius)

    def enable_filter(self, name, enabled=True):
        """Enable or disable a filter by name."""
        self.engine.set_filter_enabled(name, enabled)

    def get_active_filters(self):
        """List currently active filters."""
        return self.engine.get_active_filters()

    def process_frames(self, count=1):
        """Process N frames."""
        self.engine.run(count)

    def get_results(self):
        """Get current benchmark results."""
        return self.engine.get_benchmark_results()

    def print_results(self, results):
        """Pretty print benchmark results."""
        print(f"\n{'─'*50}")
        print(f"  Average FPS:      {results.avg_fps:.1f}")
        print(f"  Frame Time:       {results.avg_frame_time_ms:.2f} ms")
        print(f"  P95 Frame Time:   {results.p95_frame_time_ms:.2f} ms")
        print(f"  GPU Time:         {results.avg_gpu_time_ms:.2f} ms")
        print(f"  Latency:          {results.avg_latency_ms:.2f} ms")
        print(f"  GPU Utilization:  {results.gpu_utilization_pct:.1f}%")
        print(f"  Bandwidth:        {results.bandwidth_gbps:.1f} GB/s")
        print(f"  Dropped Frames:   {results.dropped_frames}")
        print(f"  Active Filters:   {', '.join(results.filter_names)}")
        print(f"{'─'*50}")


def interactive_mode(controller):
    """Interactive REPL for live parameter tweaking."""
    print("\nMetal 4 Image Pipeline - Interactive Control")
    print("Commands:")
    print("  bilateral <sigma_s> <sigma_r> <radius>")
    print("  unsharp <sigma> <amount> <threshold>")
    print("  color <exposure> <contrast> <saturation> <temperature>")
    print("  edge <threshold> <sigma> <detail_boost> <radius>")
    print("  enable <filter_name>")
    print("  disable <filter_name>")
    print("  filters        - List active filters")
    print("  bench [frames]  - Run benchmark")
    print("  run <frames>   - Process N frames")
    print("  quit           - Exit")
    print()

    while True:
        try:
            line = input("m4pipe> ").strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "bilateral" and len(parts) >= 4:
                controller.set_bilateral(float(parts[1]), float(parts[2]), int(parts[3]))
                print("  Updated bilateral filter")
            elif cmd == "unsharp" and len(parts) >= 4:
                controller.set_unsharp(float(parts[1]), float(parts[2]), float(parts[3]))
                print("  Updated unsharp mask")
            elif cmd == "color" and len(parts) >= 5:
                controller.set_color_grading(float(parts[1]), float(parts[2]),
                                              float(parts[3]), float(parts[4]))
                print("  Updated color grading")
            elif cmd == "edge" and len(parts) >= 5:
                controller.set_edge_smooth(float(parts[1]), float(parts[2]),
                                            float(parts[3]), int(parts[4]))
                print("  Updated edge smoothing")
            elif cmd == "enable" and len(parts) >= 2:
                controller.enable_filter(parts[1], True)
                print(f"  Enabled {parts[1]}")
            elif cmd == "disable" and len(parts) >= 2:
                controller.enable_filter(parts[1], False)
                print(f"  Disabled {parts[1]}")
            elif cmd == "filters":
                filters = controller.get_active_filters()
                print(f"  Active: {', '.join(filters)}")
            elif cmd == "bench":
                frames = int(parts[1]) if len(parts) > 1 else 300
                controller.process_frames(frames)
                results = controller.get_results()
                controller.print_results(results)
            elif cmd == "run" and len(parts) >= 2:
                controller.process_frames(int(parts[1]))
                print(f"  Processed {parts[1]} frames")
            else:
                print(f"  Unknown command: {line}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")

    print("Goodbye.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Metal 4 Image Pipeline Control")
    parser.add_argument("--width", type=int, default=3840, help="Image width")
    parser.add_argument("--height", type=int, default=2160, help="Image height")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark and exit")
    parser.add_argument("--frames", type=int, default=300, help="Benchmark frames")
    args = parser.parse_args()

    controller = PipelineController(args.width, args.height)
    print(f"Device: {controller.engine.get_device_name()}")

    if args.benchmark:
        results = controller.run_benchmark()
        controller.print_results(results)

        # Also run CPU baseline
        cpu_time = m4.run_cpu_baseline(args.width, args.height, 10)
        speedup = cpu_time / results.avg_frame_time_ms if results.avg_frame_time_ms > 0 else 0
        print(f"\n  CPU Baseline:     {cpu_time:.2f} ms")
        print(f"  GPU Speedup:      {speedup:.1f}x")
    else:
        interactive_mode(controller)


if __name__ == "__main__":
    main()
