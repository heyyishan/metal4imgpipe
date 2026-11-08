from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "m4pipe_py",
        ["src/python/bindings.mm"],
        include_dirs=["src", "src/core", "src/filters", "src/display"],
        extra_compile_args=["-std=c++23", "-fobjc-arc", "-O2"],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
            "-framework", "QuartzCore",
            "-framework", "IOSurface",
            "-framework", "Accelerate",
            "-framework", "AppKit",
            "-framework", "CoreGraphics",
        ],
    ),
]

setup(
    name="m4pipe",
    version="1.0.0",
    author="Metal 4 Pipeline",
    description="Metal 4 Real-Time Image Processing Pipeline",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
