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

# --- NumKong (pre-built shared library with dynamic dispatch) ---
# Build NK separately following its own docs:
#   cd ~/NumKong
#   cmake -B build_shared -DCMAKE_BUILD_TYPE=Release -DNK_BUILD_SHARED=ON -DNK_DYNAMIC_DISPATCH=1
#   cmake --build build_shared --target nk_shared --parallel
# Then pass both paths to this project:
#   cmake -B build -DNUMKONG_DIR=~/NumKong -DNUMKONG_LIB_DIR=~/NumKong/build_shared
set(NUMKONG_DIR "" CACHE PATH "Path to NumKong source tree (for headers)")
set(NUMKONG_LIB_DIR "" CACHE PATH "Path to pre-built NumKong shared library")
if(NOT NUMKONG_DIR)
    message(FATAL_ERROR "Set -DNUMKONG_DIR=/path/to/NumKong")
endif()
if(NOT NUMKONG_LIB_DIR)
    message(FATAL_ERROR "Set -DNUMKONG_LIB_DIR=/path/to/NumKong/build_shared")
endif()

add_library(nk_shared SHARED IMPORTED)
set_target_properties(nk_shared PROPERTIES
    IMPORTED_LOCATION "${NUMKONG_LIB_DIR}/libnumkong.so"
    INTERFACE_INCLUDE_DIRECTORIES "${NUMKONG_DIR}/include"
)

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
# NK_DYNAMIC_DISPATCH=1 ensures nk_dot_i8() etc. resolve to the shared library's
# dynamic dispatch symbols rather than the header's compile-time static dispatch.
# The shared library must also be built with NK_DYNAMIC_DISPATCH=1 (see instructions above).
# Note: f32 functions have an unguarded static dispatch in the NK headers, so f32
# benchmarks may use the inlined path rather than the shared library on x86.
target_compile_options(nk_bench PRIVATE -O3)
target_compile_definitions(nk_bench PRIVATE NK_DYNAMIC_DISPATCH=1)
target_link_libraries(nk_bench PRIVATE nk_shared benchmark::benchmark benchmark::benchmark_main)

if(ARCH_X86)
    target_compile_definitions(nk_bench PRIVATE ARCH_X86=1)
endif()
if(ARCH_ARM)
    target_compile_definitions(nk_bench PRIVATE ARCH_ARM=1)
endif()

# --- Direct ISA benchmark (calls Ice Lake / Haswell kernels directly) ---
add_executable(nk_bench_direct src/nk_bench_direct.cpp)
target_link_libraries(nk_bench_direct PRIVATE nk_shared benchmark::benchmark benchmark::benchmark_main)
if(ARCH_X86)
    # Need AVX-512 + VNNI for direct icelake calls (header-only, compiled into this TU)
    target_compile_options(nk_bench_direct PRIVATE -O3 -march=icelake-client)
elseif(ARCH_ARM)
    target_compile_options(nk_bench_direct PRIVATE -O3 -march=armv8.2-a+dotprod)
endif()
