# WIP: ES SimdVec Comparison

Partial benchmarks collected March-April 2026. All times in nanoseconds per operation.

**IMPORTANT:** These results were collected with GCC-14-compiled libraries.
To be re-run with Clang-21 after PR #145681 merges. Clang produces 8-12%
faster AVX-512 code due to better register allocation.

ES SimdVec benchmarked from Java via JMH. Native C kernels called via FFI
(Foreign Function & Memory API). Results include FFI call overhead.

## 1. ES Single-Pair Results

### AMD c8a — ES single-pair i8, AVX2 baseline (ns/op)

| Dims | dot | sqeuclidean |
|---|---|---|
| 384 | 11.2 | 12.1 |
| 768 | 19.8 | 21.2 |
| 1024 | 25.5 | 27.6 |
| 1536 | 37.0 | 39.3 |

### AMD c8a — ES single-pair i8, AVX-512 cascade 4/2/1 (ns/op)

| Dims | dot | sqeuclidean | cosine |
|---|---|---|---|
| 768 | 11.9 | 13.8 | 22.2 |
| 1024 | 14.1 | 16.2 | 26.0 |

### Intel c8i — ES single-pair i8, AVX2 baseline (ns/op)

| Dims | dot | sqeuclidean |
|---|---|---|
| 384 | 15.6 | 16.9 |
| 768 | 27.7 | 29.7 |
| 1024 | 34.4 | 38.8 |
| 1536 | 50.3 | 55.8 |

### Intel c8i — ES single-pair i8, AVX-512 cascade 4/2/1 (ns/op)

| Dims | dot | sqeuclidean | cosine |
|---|---|---|---|
| 768 | 19.1 | 21.7 | 37.8 |
| 1024 | 25.8 | 29.5 | 48.2 |

## 2. AVX-512 vs AVX2 Speedup

**AMD c8a — dot i8:**

| Dims | AVX2 | AVX-512 | Speedup |
|---|---|---|---|
| 384 | 11.2 | 8.4 | 1.33x |
| 768 | 19.8 | 11.6 | 1.71x |
| 1024 | 25.5 | 13.8 | 1.85x |
| 1536 | 37.0 | 18.3 | 2.02x |

**Intel c8i — dot i8:**

| Dims | AVX2 | AVX-512 | Speedup |
|---|---|---|---|
| 384 | 15.6 | 12.2 | 1.28x |
| 768 | 27.7 | 18.8 | 1.47x |
| 1024 | 34.4 | 25.4 | 1.35x |
| 1536 | 50.3 | 35.5 | 1.42x |

## 3. ES Bulk Results

ES bulk uses prefetching and batch processing. NK has no bulk operation.
`VectorScorerInt8BulkOperationBenchmark`, bulkSize=32.

### AMD c8a — ES vs NK, dot i8, 1024 dims, random access

| Dataset | ES Single (ns/vec) | ES Bulk (ns/vec) | NK Loop (ns/vec) | ES Bulk vs NK |
|---|---|---|---|---|
| 128 (L2) | 29.3 | 24.1 | 54.8 | ES 2.3x faster |
| 2500 (L3) | 37.7 | 27.1 | 70.2 | ES 2.6x faster |
| 130000 (>L3) | 139.6 | 50.6 | 174.3 | ES 3.4x faster |

### ARM c8gd — ES vs NK, dot i8, 1024 dims, random access

| Dataset | ES Single (ns/vec) | ES Bulk (ns/vec) | NK Loop (ns/vec) | ES Bulk vs NK |
|---|---|---|---|---|
| 128 (L2) | 38.9 | 20.0 | 24.4 | ES 1.2x faster |
| 2500 (L3) | 50.4 | 27.4 | 41.9 | ES 1.5x faster |
| 130000 (>L3) | 84.4 | 47.0 | 72.1 | ES 1.5x faster |

## 4. GCC vs Clang Compiler Comparison

Same source code, different compiler. Single-pair dot i8, 768 dims.

| | AMD c8a | Intel c8i |
|---|---|---|
| AVX2 baseline (gcc, no cascade) | 19.8 ns | 27.7 ns |
| AVX2 cascade (gcc) | 16.0 ns | 29.2 ns |
| AVX2 cascade (clang) | 15.5 ns | 28.5 ns |
| AVX-512 cascade (gcc) | 11.6 ns | 18.8 ns |
| AVX-512 cascade (clang) | 10.7 ns | 16.5 ns |
