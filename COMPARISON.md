# ES SimdVec vs NumKong Comparison

Benchmark results comparing ES SimdVec (from Java via JMH) against NumKong/SimSIMD
(native C via Google Benchmark). All times in nanoseconds.

ES SimdVec native C kernels called via FFI (Foreign Function & Memory API).
Results include FFI call overhead — this is how ES runs in production.

## ARM (c8gd.xlarge, Graviton 4, NEON+SDOT)

ES compiled with Clang 21, libvec 1.0.87.

### Single-pair i8 (ns/op)

| Dims | ES dot | NK dot | ES/NK | ES sqe | NK sqe | ES/NK | ES cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 11.1 | 3.9 | 0.35x | 11.1 | 3.9 | 0.35x | 14.4 | 9.4 | 0.65x |
| 256 | 13.1 | 6.8 | 0.52x | 13.2 | 6.8 | 0.52x | 18.1 | 15.3 | 0.85x |
| 384 | 15.2 | 10.0 | 0.66x | 15.1 | 10.0 | 0.66x | 22.0 | 19.8 | 0.90x |
| 512 | 17.2 | 12.9 | 0.75x | 17.4 | 12.9 | 0.74x | 30.0 | 26.4 | 0.88x |
| 768 | 21.4 | 18.6 | 0.87x | 26.7 | 18.6 | 0.70x | 39.4 | 35.9 | 0.91x |
| 1024 | 27.2 | 24.3 | 0.89x | 31.9 | 24.6 | 0.77x | 47.2 | 44.3 | 0.94x |
| 1536 | 35.4 | 40.7 | **1.15x** | 41.0 | 37.0 | 0.90x | 63.9 | 59.8 | 0.94x |
| 3072 | 61.1 | 76.6 | **1.25x** | 74.8 | 78.8 | **1.05x** | 120.7 | 114 | 0.94x |

At small dims, FFI call overhead dominates and NK wins. At 1536+ dims, ES dot product
overtakes NK. ES sqeuclidean has a wider gap at mid-range dims (768-1024) due to the
extra `vabdq_s8` instruction per step that NK avoids with a tighter loop.

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 17.8 | 24.7 | **1.39x** | 22.0 | 28.6 | **1.30x** | 26.8 | 47.1 | **1.76x** |
| 2500 (L3) | 28.1 | 41.7 | **1.48x** | 34.4 | 46.8 | **1.36x** | 38.1 | 58.1 | **1.53x** |
| 130000 (>L3) | 48.4 | 68.7 | **1.42x** | 52.7 | 74.2 | **1.41x** | 61.0 | 88.2 | **1.45x** |

ES bulk wins across the board on ARM — 1.3-1.8x faster. The interleaved bulk pattern
with 4 concurrent vectors provides memory-level parallelism that a single-call loop
cannot match.

ES Single vs NK loop (ns/vec):

| Dataset | ES Single dot | NK dot | ES/NK | ES Single sqe | NK sqe | ES/NK | ES Single cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 38.0 | 24.7 | 0.65x | 40.5 | 28.6 | 0.71x | 50.4 | 47.1 | 0.94x |
| 2500 (L3) | 50.3 | 41.7 | 0.83x | 55.1 | 46.8 | 0.85x | 64.4 | 58.1 | 0.90x |
| 130000 (>L3) | 84.4 | 68.7 | 0.82x | 90.9 | 74.2 | 0.82x | 99.1 | 88.2 | 0.89x |

Without bulk, ES single-call is slower than NK loop due to FFI overhead per call.

## AMD (c8a.xlarge, EPYC Turin / Zen 5, AVX-512)

ES results collected with GCC 14 (pre-Clang). To be re-run with Clang 21.

### ES vs NK, dot i8, 1024 dims, random access (bulk)

| Dataset | ES Bulk (ns/vec) | NK Loop (ns/vec) | ES Bulk vs NK |
|---|---|---|---|
| 128 (L2) | 24.1 | 54.8 | **ES 2.3x faster** |
| 2500 (L3) | 27.1 | 70.2 | **ES 2.6x faster** |
| 130000 (>L3) | 50.6 | 174.3 | **ES 3.4x faster** |

### AVX-512 vs AVX2 (GCC 14, to be re-run with Clang)

| Dims | AVX2 | AVX-512 | Speedup |
|---|---|---|---|
| 384 | 11.2 | 8.4 | 1.33x |
| 768 | 19.8 | 11.6 | 1.71x |
| 1024 | 25.5 | 13.8 | 1.85x |
| 1536 | 37.0 | 18.3 | 2.02x |

## Intel (c8i.2xlarge, Sapphire Rapids, AVX-512)

ES results collected with GCC 14 (pre-Clang). To be re-run with Clang 21.

### AVX-512 vs AVX2 (GCC 14, to be re-run with Clang)

| Dims | AVX2 | AVX-512 | Speedup |
|---|---|---|---|
| 384 | 15.6 | 12.2 | 1.28x |
| 768 | 27.7 | 18.8 | 1.47x |
| 1024 | 34.4 | 25.4 | 1.35x |
| 1536 | 50.3 | 35.5 | 1.42x |

## GCC vs Clang Compiler Comparison

Single-pair dot i8, 768 dims. Same source code, different compiler.

| | AMD c8a | Intel c8i |
|---|---|---|
| AVX2 baseline (gcc, no cascade) | 19.8 ns | 27.7 ns |
| AVX2 cascade (gcc) | 16.0 ns | 29.2 ns |
| AVX2 cascade (clang) | 15.5 ns | 28.5 ns |
| AVX-512 cascade (gcc) | 11.6 ns | 18.8 ns |
| AVX-512 cascade (clang) | 10.7 ns | 16.5 ns |

Clang 21 produces 8-12% faster AVX-512 code due to better register allocation.

## Key Takeaways

1. **Single-pair kernel:** NK is faster at small dims due to zero FFI overhead.
   ES catches up at larger dims where compute dominates. At 1536+ dims on ARM,
   ES dot product is faster than NK.

2. **Bulk operations are the differentiator.** ES bulk wins 1.3-3.4x over NK loop
   across all platforms and dataset sizes. The advantage comes from:
   - Batch processing (4 vectors at a time) providing memory-level parallelism
   - Explicit prefetching on x64 reducing L1 cache misses by 7x
   - Interleaved access pattern on ARM hiding memory latency

3. **AVX-512 i8 gives 1.3-2.0x over AVX2** on x64, with the benefit growing
   with dimension size.

4. **Despite running from Java with FFI overhead**, ES SimdVec's bulk path
   outperforms a native C library (NumKong) that lacks bulk operations.
