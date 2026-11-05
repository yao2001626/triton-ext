cmake_minimum_required(VERSION 3.20)
project(ascend_npu_backend LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('triton').origin).parent)"
    OUTPUT_VARIABLE TRITON_WHEEL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import importlib.metadata; print(importlib.metadata.version('triton'))"
    OUTPUT_VARIABLE TRITON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
if(NOT TRITON_VERSION STREQUAL "3.7.0")
    message(FATAL_ERROR "Ascend bindings require Triton 3.7.0; found ${TRITON_VERSION}")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 3.0.1 EXACT CONFIG REQUIRED)

set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH "Triton 3.7 LLVM ac5dc54d installation")
if(NOT EXISTS "${LLVM_INSTALL_DIR}/include/mlir/IR/Operation.h")
    message(FATAL_ERROR "Set LLVM_INSTALL_DIR to Triton 3.7's LLVM ac5dc54d headers")
endif()
find_library(TRITON_LIB NAMES triton PATHS "${TRITON_WHEEL_DIR}/_C" NO_DEFAULT_PATH REQUIRED)
add_library(triton_core SHARED IMPORTED)
set_target_properties(triton_core PROPERTIES IMPORTED_LOCATION "${TRITON_LIB}" IMPORTED_NO_SONAME TRUE)

pybind11_add_module(_ascend_ir src/ascend_ir.cpp src/ProgramGridTransform.cpp)
target_include_directories(_ascend_ir SYSTEM PRIVATE "${LLVM_INSTALL_DIR}/include")
target_compile_features(_ascend_ir PRIVATE cxx_std_17)
target_link_libraries(_ascend_ir PRIVATE triton_core)
set_target_properties(_ascend_ir PROPERTIES INSTALL_RPATH "$ORIGIN/../triton/_C")
install(TARGETS _ascend_ir LIBRARY DESTINATION triton_ascend_backend)
