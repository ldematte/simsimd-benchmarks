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
| 128 | 5.7 | 2.46 | 0.43x | 5.8 | 2.51 | 0.43x | 9.2 | 6.88 | 0.75x |
| 256 | 6.8 | 3.15 | 0.46x | 7.0 | 3.64 | 0.52x | 11.3 | 8.95 | 0.79x |
| 384 | 7.7 | 4.08 | 0.53x | 7.8 | 5.02 | 0.64x | 13.3 | 11.1 | 0.83x |
| 512 | 8.6 | 4.96 | 0.58x | 9.1 | 6.32 | 0.69x | 16.8 | 13.2 | 0.79x |
| 768 | 10.7 | 6.94 | 0.65x | 10.9 | 9.51 | 0.87x | 20.9 | 18.3 | 0.88x |
| 1024 | 12.6 | 8.67 | 0.69x | 13.1 | 12.2 | 0.93x | 25.2 | 22.9 | 0.91x |
| 1536 | 16.7 | 14.8 | 0.89x | 16.8 | 19.9 | **1.18x** | 33.2 | 31.8 | 0.96x |
| 3072 | 30.0 | 29.2 | 0.97x | 29.8 | 37.7 | **1.27x** | 64.9 | 60.5 | 0.93x |

NK wins at small-to-medium dimensions on all operations thanks to zero FFI overhead
and AVX-512 VNNI (XOR+DPBUSD, 64 bytes/iter). ES catches up at larger dimensions where
its 4-way cascade unrolling amortizes the ~5 ns FFI cost. ES wins on sqeuclidean at
1536+ dims. Both libraries use AVX-512; NK uses VNNI DPBUSD while ES uses
sign-extension + madd_epi16 with cascade unrolling.

At small dimensions (128-512), the gap is almost entirely call overhead: NK goes through
a shared-library dispatch (~1 ns), ES goes through FFI (~5 ns). At 1024 dims, the kernel
times are nearly identical (~7.6 ns ES vs ~7.7 ns NK) — the entire difference is call
overhead. ES's cascade unrolling only starts to show a kernel-level advantage at 1536+
dims, and for sqeuclidean it matters earlier because NK's kernel does 2x the work per
iteration (unpack to i16 + two DPWSSDs vs ES's single madd_epi16).

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 11.2 | 13.8 | **1.23x** | 14.2 | 17.8 | **1.25x** | 12.7 | 24.2 | **1.91x** |
| 2500 (L3) | 15.5 | 20.4 | **1.32x** | 17.4 | 23.8 | **1.37x** | 16.5 | 30.4 | **1.84x** |
| 130000 (>L3) | 39.9 | 62.5 | **1.57x** | 42.2 | 83.1 | **1.97x** | 37.7 | 114.6 | **3.04x** |

ES bulk wins across the board on AMD, with the advantage growing at larger dataset
sizes. The combination of batch processing + explicit prefetching hides memory latency
that a loop of single-pair calls cannot.

## Intel (c8i.2xlarge, Granite Rapids, AVX-512)

ES compiled with Clang 21, libvec 1.0.87 (AVX-512 i8 kernels with cascade unrolling).

### Single-pair i8 (ns/op)

| Dims | ES dot | NK dot | ES/NK | ES sqe | NK sqe | ES/NK |
|---|---|---|---|---|---|---|
| 128 | 8.4 | 3.88 | 0.46x | 8.7 | 3.80 | 0.44x |
| 256 | 9.9 | 5.56 | 0.56x | 10.2 | 6.40 | 0.63x |
| 384 | 11.4 | 6.82 | 0.60x | 11.8 | 8.87 | 0.75x |
| 512 | 12.8 | 8.49 | 0.66x | 15.0 | 11.5 | 0.77x |
| 768 | 16.7 | 14.0 | 0.84x | 18.7 | 17.1 | 0.91x |
| 1024 | 21.1 | 15.9 | 0.75x | 25.1 | 21.5 | 0.86x |
| 1536 | 30.3 | 26.1 | 0.86x | 35.3 | 31.7 | 0.90x |
| 3072 | 54.2 | 51.6 | 0.95x | 64.4 | 66.2 | **1.03x** |

Angular not yet collected on Intel. Similar pattern to AMD — NK wins at all dimensions
for dot, ES only overtakes on sqeuclidean at 3072.

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 22.4 | 18.6 | 0.83x | 28.5 | 25.2 | 0.88x | 21.1 | 48.7 | **2.31x** |
| 2500 (L3) | 38.8 | 38.4 | 0.99x | 42.2 | 52.0 | **1.23x** | 38.2 | 70.8 | **1.85x** |
| 130000 (in L3) | 58.1 | 51.9 | 0.89x | 64.4 | 67.2 | **1.04x** | 55.2 | 85.7 | **1.55x** |

On Intel, ES bulk shows less advantage than on AMD — NK's dot loop matches or beats
ES bulk across dataset sizes. The 480 MB L3 holds the entire 130k dataset, reducing
the prefetching benefit that drives ES bulk's advantage on AMD. ES bulk still wins
on cosine and sqeuclidean. The ES bulk absolute times on Intel are notably higher than
on AMD (e.g. 22.4 vs 11.2 ns/vec at L2 for dot) — this gap needs further investigation.

## ARM (c8gd.xlarge, Graviton 4, NEON+SDOT)

ES compiled with Clang 21, libvec 1.0.87.

### Single-pair i8 (ns/op)

| Dims | ES dot | NK dot | ES/NK | ES sqe | NK sqe | ES/NK | ES cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 11.1 | 5.01 | 0.45x | 11.1 | 6.01 | 0.54x | 14.4 | 8.24 | 0.57x |
| 256 | 13.1 | 7.87 | 0.60x | 13.2 | 10.2 | 0.77x | 18.1 | 13.2 | 0.73x |
| 384 | 15.2 | 11.1 | 0.73x | 15.1 | 14.4 | 0.95x | 22.0 | 18.2 | 0.83x |
| 512 | 17.2 | 15.5 | 0.90x | 17.4 | 16.7 | 0.96x | 30.0 | 23.5 | 0.78x |
| 768 | 21.4 | 24.6 | **1.15x** | 26.7 | 24.1 | 0.90x | 39.4 | 33.8 | 0.86x |
| 1024 | 27.2 | 32.5 | **1.20x** | 31.9 | 32.2 | **1.01x** | 47.2 | 42.4 | 0.90x |
| 1536 | 35.4 | 49.3 | **1.39x** | 41.0 | 49.0 | **1.20x** | 63.9 | 60.0 | 0.94x |
| 3072 | 61.1 | 111 | **1.82x** | 74.8 | 107 | **1.43x** | 120.7 | 116 | 0.96x |

NK wins at small dims (lower call overhead). ES overtakes on dot at 768+ dims and on
sqeuclidean at 1024+ dims. The shared library dispatch overhead on ARM (~5-8 ns) is
larger than on x86 (~1 ns), which shifts the crossover point. Both libraries use
NEON+SDOT, so at large dims ES's cascade unrolling gives a genuine kernel advantage
(up to 1.8x at 3072).

### Multi-vector i8, 1024 dims, random access, bulkSize=32

ES Bulk vs NK loop (ns/vec):

| Dataset | ES Bulk dot | NK dot | ES/NK | ES Bulk sqe | NK sqe | ES/NK | ES Bulk cos | NK cos | ES/NK |
|---|---|---|---|---|---|---|---|---|---|
| 128 (L2) | 17.8 | 32.8 | **1.84x** | 22.0 | 35.5 | **1.61x** | 26.8 | 48.1 | **1.80x** |
| 2500 (L3) | 28.1 | 48.1 | **1.71x** | 34.4 | 48.8 | **1.42x** | 38.1 | 62.1 | **1.63x** |
| 130000 (>L3) | 48.4 | 89.6 | **1.85x** | 52.7 | 93.7 | **1.78x** | 61.0 | 99.9 | **1.64x** |

ES bulk wins 1.4-1.9x across all ops and dataset sizes. The interleaved bulk pattern
with 4 concurrent vectors provides memory-level parallelism that a single-call loop
cannot match.


## Key Takeaways

1. **On x86 single-pair, NK is faster at small-to-medium dimensions** thanks to
   AVX-512 VNNI (XOR+DPBUSD) and lower call overhead (~1 ns shared lib dispatch vs
   ~5 ns ES FFI). ES catches up at larger dimensions where cascade unrolling amortizes
   the FFI cost. At 1024 dims the kernel times are nearly identical — the gap is
   almost entirely call overhead.

2. **On ARM single-pair, NK wins at small dims, ES wins at 768+.** The shared library
   dispatch overhead on ARM (~5-8 ns) is larger than on x86, shifting the crossover.
   At large dims, ES's cascade unrolling gives a genuine kernel advantage (up to 1.8x
   at 3072).

3. **Bulk operations are where ES consistently wins**: on AMD 1.2-3x, on ARM 1.4-1.9x.
   NK has no bulk API, so its loop of single-pair calls cannot hide memory latency. On
   Intel, the 480 MB L3 holds the dataset, so NK's single-pair loop keeps up on dot —
   but ES bulk still wins on cosine and sqeuclidean.
