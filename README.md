# NumKong (SimSIMD) Distance Kernel Benchmarks

Google Benchmark harness for NumKong (formerly SimSIMD) low-level SIMD
distance functions. Part of the ES SimdVec comparison suite — measures
NumKong natively in C++ to compare against ES SimdVec (benchmarked from
Java via JMH).

## What is benchmarked

| Level | Benchmark                          | NumKong function       | Comparable ES SimdVec API                        |
|-------|------------------------------------|------------------------|--------------------------------------------------|
| 1     | `BM_dot_i8`                       | `nk_dot_i8`           | `VectorScorerInt8OperationBenchmark` (DOT)       |
| 1     | `BM_sqeuclidean_i8`               | `nk_sqeuclidean_i8`   | `VectorScorerInt8OperationBenchmark` (EUCLIDEAN)  |
| 1     | `BM_angular_i8`                   | `nk_angular_i8`       | `VectorScorerInt8OperationBenchmark` (COSINE)     |
| 1     | `BM_dot_f32`                      | `nk_dot_f32`          | `VectorScorerFloat32OperationBenchmark` (DOT)     |
| 1     | `BM_sqeuclidean_f32`              | `nk_sqeuclidean_f32`  | `VectorScorerFloat32OperationBenchmark` (EUCLIDEAN)|
| 2     | `BM_multi_dot_i8_sequential`      | `nk_dot_i8` (loop)    | `VectorScorerInt8BulkOperationBenchmark` (seq)    |
| 2     | `BM_multi_dot_i8_random`          | `nk_dot_i8` (loop)    | `VectorScorerInt8BulkOperationBenchmark` (random) |
| 2     | `BM_multi_sqeuclidean_i8_*`       | `nk_sqeuclidean_i8` (loop) | same, EUCLIDEAN                              |
| 2     | `BM_multi_angular_i8_*`           | `nk_angular_i8` (loop)| same, COSINE                                     |

**Level 1** — Single-vector kernel throughput (one pair, ns/op).
Dimensions: 128, 256, 384, 512, 768, 1024, 1536, 3072.

**Level 2** — Multi-vector scoring (one query vs many dataset vectors).
NumKong has no native bulk operation, so this loops calling single-pair
functions — directly comparable to ES SimdVec's `scoreMultipleRandom`.
Dataset sizes: 128, 2500, 130000 vectors at 1024 dims.

**Note:** f32 benchmarks are included for reference but we will not use them for comparison. NumKong accumulates f32 dot products in f64 (for precision),
making them ~9x slower than ES SimdVec's f32 path. Different design goals, would not be fair to compare them.

## Dependencies (Ubuntu)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git
```

- **build-essential** — GCC/G++, make
- **cmake** — build system (>= 3.14 required)

Google Benchmark is fetched automatically via CMake FetchContent.
NumKong is built as a shared library from source.

## Clone NumKong

```bash
git clone --depth 1 https://github.com/ashvardanian/NumKong.git /path/to/NumKong
```

### ARM build workaround

GCC 14 on Graviton 4 fails with `always_inline` target mismatch in
NumKong's headers. Two fixes needed:

```bash
# Remove always_inline from NK_INTERNAL
sed -i 's/#define NK_INTERNAL __attribute__((always_inline)) inline static/#define NK_INTERNAL inline static/' \
  /path/to/NumKong/include/numkong/types.h

# Disable LTO (also fails due to target mismatches)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DNUMKONG_DIR=/path/to/NumKong \
  -DCMAKE_C_FLAGS="-fno-lto"
```

## Build

```bash
cd /path/to/simsimd-benchmarks
cmake -B build \
    -DNUMKONG_DIR=/path/to/NumKong \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On ARM with the workaround:

```bash
cmake -B build \
    -DNUMKONG_DIR=/path/to/NumKong \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fno-lto"
cmake --build build
```

## Run

### List available benchmarks

```bash
./build/nk_bench --benchmark_list_tests
```

### Full benchmarks

```bash
./build/nk_bench --benchmark_repetitions=5
```

### Run a subset

Single-pair i8 only:

```bash
./build/nk_bench --benchmark_filter="BM_dot_i8|BM_sqeuclidean_i8|BM_angular_i8"
```

Only 1024 dimensions:

```bash
./build/nk_bench --benchmark_filter=".*1024.*"
```

Multi-vector only:

```bash
./build/nk_bench --benchmark_filter="BM_multi_"
```

Output as JSON (for merging with JMH results):

```bash
./build/nk_bench --benchmark_repetitions=5 \
    --benchmark_out=nk_results.json \
    --benchmark_out_format=json
```

## Reducing variance

For stable results, pin to a single CPU core and disable frequency scaling:

```bash
# x86: disable turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null
# or for AMD:
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost 2>/dev/null

# Pin to core 0
taskset -c 0 ./build/nk_bench --benchmark_repetitions=5
```

On ARM (Graviton), frequency scaling is typically fixed, but you can
still pin:

```bash
taskset -c 0 ./build/nk_bench --benchmark_repetitions=5
```

## Project structure

```
simsimd-benchmarks/
├── CMakeLists.txt          # Build config, fetches Google Benchmark, links NumKong
├── README.md               # This file
├── RESULTS.md              # Collected benchmark results
└── src/
    └── nk_bench.cpp        # Google Benchmark harness (Level 1 + 2)
```
