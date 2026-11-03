cmake_minimum_required(VERSION 3.20)
project(ascend_npu_backend LANGUAGES NONE)

# This initial scaffold ships Python backend discovery only. When the Ascend
# passes are ported, configure_triton_extension() and a native plugin target
# belong here, following backend/AppleGPU/CMakeLists.txt.
