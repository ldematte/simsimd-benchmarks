cmake_minimum_required(VERSION 3.14)
project(numkong_bench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# --- Google Benchmark via FetchContent ---
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# --- NumKong (shared library with dynamic dispatch) ---
# Build the shared library so benchmark calls go through a real function call,
# matching how a production consumer would use the library. This is fair to
# compare against ES SimdVec which goes through FFM from Java.
set(NUMKONG_DIR "" CACHE PATH "Path to NumKong source tree")
if(NOT NUMKONG_DIR)
    message(FATAL_ERROR "Set -DNUMKONG_DIR=/path/to/NumKong")
endif()

set(NK_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(NK_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(NK_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(${NUMKONG_DIR} ${CMAKE_BINARY_DIR}/numkong)

# --- Detect architecture ---
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ARCH_ARM TRUE)
else()
    set(ARCH_X86 TRUE)
endif()

# --- Benchmark executable ---
# Links against the NumKong shared library (nk_shared). At runtime, dynamic
# dispatch selects the best ISA via nk_capabilities(). Each kernel call goes
# through a real shared-library function call — no inlining.
add_executable(nk_bench src/nk_bench.cpp)
target_compile_options(nk_bench PRIVATE -O3)
target_link_libraries(nk_bench PRIVATE nk_shared benchmark::benchmark benchmark::benchmark_main)

if(ARCH_X86)
    target_compile_definitions(nk_bench PRIVATE ARCH_X86=1)
endif()
if(ARCH_ARM)
    target_compile_definitions(nk_bench PRIVATE ARCH_ARM=1)
endif()
