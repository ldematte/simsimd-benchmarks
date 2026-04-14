# NumKong (SimSIMD) Benchmark Results

Collected April 2026. All times in nanoseconds per operation (ns/op).

NumKong compiled as shared library with dynamic dispatch (`NK_BUILD_SHARED=ON`,
`NK_DYNAMIC_DISPATCH=1`). Each call goes through a real shared-library function call
(indirect branch through function pointer table), matching the overhead of a
production library consumer.

## Hardware

| Instance | CPU | ISA | L1D | L2 | L3 |
|---|---|---|---|---|---|
| AMD c8a.xlarge | EPYC Turin (Zen 5) | AVX2 + AVX-512 + VNNI | 48 KB | 1 MB | 16 MB |
| ARM c8gd.xlarge | Graviton 4 (Neoverse V2) | NEON + SDOT | 64 KB | 2 MB | 36 MB |
| Intel c8i.2xlarge | Granite Rapids (Xeon 6975P-C) | AVX2 + AVX-512 + VNNI | 48 KB | 2 MB | 480 MB |

## 1. Single-Pair Results

### AMD c8a — i8 (ns/op)

| Dims | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 | 2.46 | 2.51 | 6.88 |
| 256 | 3.15 | 3.64 | 8.95 |
| 384 | 4.08 | 5.02 | 11.1 |
| 512 | 4.96 | 6.32 | 13.2 |
| 768 | 6.94 | 9.51 | 18.3 |
| 1024 | 8.67 | 12.2 | 22.9 |
| 1536 | 14.8 | 19.9 | 31.8 |
| 3072 | 29.2 | 37.7 | 60.5 |

### ARM c8gd — i8 (ns/op)

| Dims | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 | 5.01 | 6.01 | 8.24 |
| 256 | 7.87 | 10.2 | 13.2 |
| 384 | 11.1 | 14.4 | 18.2 |
| 512 | 15.5 | 16.7 | 23.5 |
| 768 | 24.6 | 24.1 | 33.8 |
| 1024 | 32.5 | 32.2 | 42.4 |
| 1536 | 49.3 | 49.0 | 60.0 |
| 3072 | 111 | 107 | 116 |

### Intel c8i — i8 (ns/op)

| Dims | dot | sqeuclidean |
|---|---|---|
| 128 | 3.88 | 3.80 |
| 256 | 5.56 | 6.40 |
| 384 | 6.82 | 8.87 |
| 512 | 8.49 | 11.5 |
| 768 | 14.0 | 17.1 |
| 1024 | 15.9 | 21.5 |
| 1536 | 26.1 | 31.7 |
| 3072 | 51.6 | 66.2 |

### Observations

- **AMD i8 dot uses AVX-512 VNNI** — NumKong's Ice Lake kernel uses the XOR+DPBUSD
  algebraic trick (convert signed to unsigned via XOR 0x80, compute, subtract correction
  term), processing 64 bytes/iteration. Single accumulator, no unrolling.
- **ARM dot and sqeuclidean are nearly identical** — NumKong uses `vabdq_s8 + vdotq_u32`
  for sqeuclidean (same throughput as dot product via the SDOT instruction).
- **AMD sqeuclidean is ~1.4x slower than dot** — the Ice Lake sqeuclidean kernel needs
  two unpack+DPWSSD operations per 64 input bytes (widening to i16 for squaring).
- **Angular is ~2.5x slower than dot** (3 concurrent accumulators plus final sqrt).
- **f32 excluded from comparison** — NumKong accumulates f32 in f64 for precision
  (documented design choice: "Mixed precision by default"). The f64 widening cost
  at 1024 dims (NK vs ES f32 dot): AMD ~89 ns vs ~28 ns (3.2x), ARM ~373 ns vs
  ~73 ns (5.1x). ARM penalty is larger because NEON processes only 2×f64 per register
  vs 4×f32, compounding the lane disadvantage with scalar reduction overhead.
  This is a deliberate precision/throughput tradeoff, not a performance deficiency.

## 2. Multi-Vector Results

Loop calling NumKong single-pair functions over a contiguous dataset with random
access pattern. NumKong has no native bulk/batch operation — this measures the
realistic overhead of calling a library without bulk support in a hot loop.

Dataset sizes tuned for cache hierarchy:
- 128 vectors (128 KB): overflows L1, fits in L2
- 2500 vectors (2.5 MB): overflows L2 on all platforms, fits in L3
- 130000 vectors (~127 MB): overflows L3 on AMD/ARM; fits in L3 on Intel (480 MB)

### AMD c8a — i8, 1024 dims, random access (ns/vec)

| Dataset | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 (L2) | 13.8 | 17.8 | 24.2 |
| 2500 (L3) | 20.4 | 23.8 | 30.4 |
| 130000 (>L3) | 62.5 | 83.1 | 114.6 |

### ARM c8gd — i8, 1024 dims, random access (ns/vec)

| Dataset | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 (L2) | 32.8 | 35.5 | 48.1 |
| 2500 (L3) | 48.1 | 48.8 | 62.1 |
| 130000 (>L3) | 89.6 | 93.7 | 99.9 |

### Intel c8i — i8, 1024 dims, random access (ns/vec)

| Dataset | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 (L2) | 18.6 | 25.2 | 48.7 |
| 2500 (L3) | 38.4 | 52.0 | 70.8 |
| 130000 (in L3) | 51.9 | 67.2 | 85.7 |

### Observations

- **Random access penalty grows with dataset size**: on AMD, dot i8 goes from 14 ns
  (L2) to 63 ns (beyond L3) — 4.5x slower. On ARM, 33 ns to 90 ns — 2.7x slower.
- **Intel 130k barely worse than 2500** because 127 MB fits in the 480 MB L3 cache.
- **The random access penalty is where bulk operations with prefetching make a
  difference** — a library that can batch multiple vector comparisons and prefetch
  ahead can significantly reduce the cache miss cost.
