# ES SimdVec vs NumKong Comparison

Benchmark results comparing ES SimdVec (from Java via JMH) against NumKong/SimSIMD
(native C/C++ via Google Benchmark). All times in nanoseconds.

ES SimdVec native C kernels called via FFI (Foreign Function & Memory API).
Results include FFI call overhead — this is how ES runs in production.

## FFM Downcall overhead measurements

In order to interpret numbers correctly, we designed a couple of "experiments"/benchmarks
to assess the cost of FFI/FFM. Results include FFI call overhead, as this is how ES runs in production,
but for a complete picture we wanted to document which portion of the results may be 
due to the call overhead.

### Direct measurement: confined vs shared arena

For the first experiment, we designed a benchmark which calls one of the native kernels (`vec_dotf32`)
 via FFM with different arena types. The `size=1` case isolates the pure FFM overhead (a 1-float dot
product is essentially a no-op on the native side). For x64, the numbers are:

| Benchmark          | size=1 (ns)       | size=768 (ns)      | size=1024 (ns)     |
|--------------------|-------------------|--------------------|--------------------|
| confinedSegment    | **5.072 ± 0.147** | 24.420 ± 2.790     | 32.219 ± 4.182     |
| sharedSegment      | **11.599 ± 0.011**| 25.640 ± 3.362     | 35.396 ± 3.865     |
| delta (shared−conf)| **+6.527 ns**     | ~+1.2 ns           | ~+3.2 ns           |


Some notes:

- **Confined arena downcall overhead: ~5 ns.** This is the pure FFM cost:
   downcall stub entry, argument marshalling (MemorySegment → raw pointer),
   native `call`, and return. Note the tiny error bars (± 0.147 ns) — there's
   no data-dependent variability at size=1.

- **Shared arena downcall overhead: ~11.6 ns.** Over 2× the confined cost.
   The extra **~6.5 ns** is the shared arena liveness check — a
   `lock cmpxchg` (CAS) the JVM emits on each downcall with a shared segment
   to ensure the arena hasn't been closed by another thread. Even uncontended,
   this is a full memory barrier on x86.
   
 At ~3.7 GHz base, 6.5 ns ≈ 24 cycles. A lock cmpxchg itself is typically 15-20 cycles on Zen 5 (the memory barrier is
  the expensive part — it has to drain the store buffer and ensure global visibility). The remaining few cycles are likely
  the JVM's arena liveness check logic wrapping the CAS: loading the state, branching on the result, etc.
  For comparison, the entire confined downcall (5 ns ≈ 18.5 cycles) fits in fewer cycles than just the shared segment's CAS
  overhead alone.
  

The ratios are almost identical on ARM (Graviton 4):

| Benchmark          | size=1 (ns)        | size=768 (ns)      | size=1024 (ns)     |
|--------------------|--------------------|--------------------|---------------------|
| confinedSegment    | **9.898 ± 0.109**  | 57.506 ± 1.022     | 74.570 ± 0.546      |
| sharedSegment      | **22.978 ± 0.037** | 71.748 ± 0.879     | 86.651 ± 1.037      |
| delta (shared−conf)| **+13.08 ns**      | +14.24 ns          | +12.08 ns           |


- ARM (Graviton 4) vector scoring times are between 2-2.8x higher than x64 (AMD Zen5)
- The confined baseline (~10 ns vs ~5 ns) is also ~2× — JVM downcall stub cost
  scales roughly with cycle time (Graviton 4 runs at ~3.5 GHz vs Zen 5 ~3.7 GHz,
  but the difference is largely in the stub's code path length on AArch64 vs x86).
- The shared arena CAS penalty (~13 ns vs ~6.5 ns) is also ~2× — the ARM
  `ldxr`/`stxr` (LL/SC) atomic pair seems more expensive than x86's
  `lock cmpxchg`, especially for the full-barrier semantics required.


On both processors, **at larger sizes, the delta is masked by noise.** The load bandwidth variability overwhelms the fixed difference. But the overhead is still there.

### Breakdown of the total cost (confined arena path)

| Component               | Cost      | Notes                                     |
|-------------------------|-----------|-------------------------------------------|
| FFM downcall stub       | ~5 ns     | Measured directly (size=1)                |
| Horizontal reduce       | ~3–4 ns   | 8 instructions: extract/add/shuffle chain |
| L1 load bandwidth       | ~13–26 ns | 2 vectors × dims × 4 B, at 128 B/cycle   |
| FMA compute             | ~3–6 ns   | Fully overlapped with loads (free)        |
| **Total (768 dims)**    | **~24 ns**| Matches measured 24.4 ns                  |
| **Total (1024 dims)**   | **~30 ns**| Matches measured 32.2 ns (within error)   |


For comparison, a native -> native call (as you might expect in the NK benchmarks), a single score function like `vec_dotf32_2` (3 arguments, all fitting in registers, no stack spill) is a bare call/ret pair on x86-64, which takes a total of ~2-3 cycles ≈ <1 ns. 
So the FFM confined overhead of ~5 ns (~18 cycles) is spending ~15-16 cycles on top of the raw call/ret. The raw native call itself is essentially free by comparison.

## AMD (c8a.xlarge, EPYC Turin / Zen 5, AVX-512)

ES compiled with Clang 21, libvec 1.0.87 (AVX-512 i8 kernels with cascade unrolling).

### Single-pair i8 (ns/op)

| Dims | ES dot | NK dot | ES/NK | ES sqe | NK sqe | ES/NK | ES cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 5.7 | 6.1 | **1.07x** | 5.8 | 10.6 | **1.83x** | 9.2 | 16.3 | **1.77x** |
| 256 | 6.8 | 11.7 | **1.72x** | 7.0 | 21.6 | **3.09x** | 11.3 | 25.8 | **2.28x** |
| 384 | 7.7 | 17.2 | **2.23x** | 7.8 | 33.9 | **4.35x** | 13.3 | 35.6 | **2.68x** |
| 512 | 8.6 | 23.4 | **2.72x** | 9.1 | 45.7 | **5.02x** | 16.8 | 45.0 | **2.68x** |
| 768 | 10.7 | 36.6 | **3.42x** | 10.9 | 73.1 | **6.71x** | 20.9 | 63.6 | **3.04x** |
| 1024 | 12.6 | 49.9 | **3.96x** | 13.1 | 104 | **7.94x** | 25.2 | 81.7 | **3.24x** |
| 1536 | 16.7 | 80.5 | **4.82x** | 16.8 | 165 | **9.82x** | 33.2 | 118 | **3.55x** |
| 3072 | 30.0 | 179 | **5.97x** | 29.8 | 352 | **11.81x** | 64.9 | 229 | **3.53x** |

ES wins at every dimension, even at 128 where FFI overhead is a significant fraction.
The sqeuclidean advantage is especially large (up to **12x**) because ES uses `abs_epi8 +
maddubs` on AVX-512 (64 bytes/iter) while NK uses sign-extension (32 bytes/iter for the
same operation).

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 11.2 | 55.3 | **4.94x** | 14.2 | 104.7 | **7.37x** | 12.7 | 78.9 | **6.21x** |
| 2500 (L3) | 15.5 | 69.6 | **4.49x** | 17.4 | 116.0 | **6.67x** | 16.5 | 86.5 | **5.24x** |
| 130000 (>L3) | 39.9 | 174.2 | **4.37x** | 42.2 | 218.0 | **5.17x** | 37.7 | 196.4 | **5.21x** |

Bulk operations give good benefits on AMD; at 1024 dims they maintain the same advantage over NK for square distance and dot product,
and give a good 2x boost to cosine.
Prefetching + batch processing help keeping timings down despite cache misses: ES Bulk at high dataset sizes (>L3) is 
**faster than single-pair NK** (L1), with 39.9 ns/vec vs 49.9 ns/vec.

## Intel (c8i.2xlarge, Sapphire Rapids, AVX-512)

ES compiled with Clang 21, libvec 1.0.87 (AVX-512 i8 kernels with cascade unrolling).

### Single-pair i8 (ns/op)

| Dims | ES dot | NK dot | ES/NK | ES sqe | NK sqe | ES/NK | ES cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 8.4 | 7.7 | 0.92x | 8.7 | 13.7 | **1.57x** | 12.1 | 21.1 | **1.74x** |
| 256 | 9.9 | 14.6 | **1.47x** | 10.2 | 26.7 | **2.62x** | 17.2 | 35.6 | **2.07x** |
| 384 | 11.4 | 21.6 | **1.89x** | 11.8 | 39.8 | **3.37x** | 20.2 | 51.3 | **2.54x** |
| 512 | 12.8 | 31.2 | **2.44x** | 15.0 | 53.1 | **3.54x** | 24.3 | 64.7 | **2.66x** |
| 768 | 16.7 | 46.9 | **2.81x** | 18.7 | 81.1 | **4.34x** | 35.9 | 91.7 | **2.55x** |
| 1024 | 21.1 | 61.2 | **2.90x** | 25.1 | 107 | **4.26x** | 45.5 | 118 | **2.59x** |
| 1536 | 30.3 | 89.6 | **2.96x** | 35.3 | 160 | **4.53x** | 60.4 | 170 | **2.81x** |
| 3072 | 54.2 | 174 | **3.21x** | 64.4 | 318 | **4.94x** | 114 | 329 | **2.89x** |

Consistent wins across all dims and operations. Smaller advantage than AMD (3-5x vs 4-12x), likely 
due to Sapphire Rapids' AVX-512 frequency throttling.

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 22.4 | 62.5 | **2.79x** | 28.5 | 108.2 | **3.80x** | 21.1 | 110.3 | **5.23x** |
| 2500 (L3) | 38.8 | 88.0 | **2.27x** | 42.2 | 130.2 | **3.09x** | 38.2 | 131.2 | **3.44x** |
| 130000 (in L3!) | 58.1 | 99.9 | **1.72x** | 64.4 | 141.6 | **2.20x** | 55.2 | 140.5 | **2.54x** |

ES bulk wins on Intel too — **1.7-5.2x faster**. The advantage is smaller at 130k
because Intel's 480 MB L3 cache holds the entire dataset, reducing the prefetching benefit.

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
overtakes NK. On ARM both ES and NK use NEON+SDOT, so the kernel compute is similar and the comparison reflects mainly the FFI overhead.

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


## Key Takeaways

1. **On x86 with AVX-512, ES SimdVec dominates at every dimension.** Even at 128 dims
   with FFI overhead, ES is faster on AMD. The advantage grows with dimension size:
   up to **6x for dot, 12x for sqeuclidean, 3.5x for cosine** on AMD at 3072 dims.

2. **On ARM, single-pair kernel performance is close.** Both ES and NK use NEON+SDOT.
   NK wins at small dims (zero FFI overhead), ES catches up at 1536+ dims. The gap
   is primarily FFI call overhead (~5-11 ns per call).

3. **Bulk operations are worth it**: on ARM, ES bulk wins 1.3-1.8x over NK loop
   across all dataset sizes. The advantage comes from batch processing (4 vectors at a
   time) providing memory-level parallelism via the interleaved access pattern. 
   On x86, ES bulk is 4-7x faster than NK loop on AMD, and 1.7-5.2x faster on Intel. The combination of AVX-512 kernels +
   explicit prefetching + batch processing makes ES bulk operations **with cache misses** faster than NK single-pair scoring (no cache misses); for dot product, ES SimdVec takes 39.9 ns/vec (130k random, DRAM latency) vs NumKong 49.9 ns/vec (single-pair, in L1): ES with cold data beats NK with hot data.

Despite running from Java with FFI overhead, ES SimdVec's optimized kernels (AVX-512, cascade unrolling, bulk operations) outperform a native C library (NumKong) that lacks these optimizations.
