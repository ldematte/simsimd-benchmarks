# NumKong (SimSIMD) Benchmark Results

Collected April 2026. All times in nanoseconds per operation (ns/op).

NumKong compiled as shared library with dynamic dispatch (`NK_BUILD_SHARED=ON`).
Each call goes through a real shared-library function call (indirect branch through
function pointer table), matching the overhead of a production library consumer.

## Hardware

| Instance | CPU | ISA | L1D | L2 | L3 |
|---|---|---|---|---|---|
| AMD c8a.xlarge | EPYC Turin (Zen 5) | AVX2 + AVX-512 | 48 KB | 1 MB | 16 MB |
| ARM c8gd.xlarge | Graviton 4 (Neoverse V2) | NEON + SDOT | 64 KB | 2 MB | 36 MB |
| Intel c8i.2xlarge | Sapphire Rapids | AVX2 + AVX-512 | 48 KB | 2 MB | 480 MB |

## 1. Single-Pair Results

### AMD c8a — i8 (ns/op)

| Dims | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 | 6.1 | 10.6 | 16.3 |
| 256 | 11.7 | 21.6 | 25.8 |
| 384 | 17.2 | 33.9 | 35.6 |
| 512 | 23.4 | 45.7 | 45.0 |
| 768 | 36.6 | 73.1 | 63.6 |
| 1024 | 49.9 | 104 | 81.7 |
| 1536 | 80.5 | 165 | 118 |
| 3072 | 179 | 352 | 229 |

### ARM c8gd — i8 (ns/op)

| Dims | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 | 3.9 | 3.9 | 9.4 |
| 256 | 6.8 | 6.8 | 15.3 |
| 384 | 10.0 | 10.0 | 19.8 |
| 512 | 12.9 | 12.9 | 26.4 |
| 768 | 18.6 | 18.6 | 35.9 |
| 1024 | 24.3 | 24.6 | 44.3 |
| 1536 | 40.7 | 37.0 | 59.8 |
| 3072 | 76.6 | 78.8 | 114 |

### Intel c8i — i8 (ns/op)

| Dims | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 | 7.7 | 13.7 | 21.1 |
| 256 | 14.6 | 26.7 | 35.6 |
| 384 | 21.6 | 39.8 | 51.3 |
| 512 | 31.2 | 53.1 | 64.7 |
| 768 | 46.9 | 81.1 | 91.7 |
| 1024 | 61.2 | 107 | 118 |
| 1536 | 89.6 | 160 | 170 |
| 3072 | 174 | 318 | 329 |

### Observations

- **ARM dot and sqeuclidean are nearly identical** — NumKong uses `vabdq_s8 + vdotq_u32`
  for sqeuclidean (same throughput as dot product via the SDOT instruction).
- **AMD/Intel sqeuclidean is ~2x slower than dot** — NumKong uses sign-extension to
  16-bit for both operations on x86, no `maddubs` optimization for sqeuclidean.
- **Angular is ~1.5-2x slower than dot** on all platforms (3 concurrent accumulators:
  sum, a_norm, b_norm).
- **f32 results excluded** — NumKong accumulates f32 dot products in f64 for precision
  (documented design choice: "Mixed precision by default"). This makes f32 operations
  ~9x slower than a f32-accumulating implementation. Different design goal from
  vector search (where ranking order matters, not exact values).

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
| 128 (L2) | 55.3 | 104.7 | 78.9 |
| 2500 (L3) | 69.6 | 116.0 | 86.5 |
| 130000 (>L3) | 174.2 | 218.0 | 196.4 |

### ARM c8gd — i8, 1024 dims, random access (ns/vec)

| Dataset | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 (L2) | 24.7 | 28.6 | 47.1 |
| 2500 (L3) | 41.7 | 46.8 | 58.1 |
| 130000 (>L3) | 68.7 | 74.2 | 88.2 |

### Intel c8i — i8, 1024 dims, random access (ns/vec)

| Dataset | dot | sqeuclidean | angular |
|---|---|---|---|
| 128 (L2) | 62.5 | 108.2 | 110.3 |
| 2500 (L3) | 88.0 | 130.2 | 131.2 |
| 130000 (in L3!) | 99.9 | 141.6 | 140.5 |

### Observations

- **Random access penalty grows with dataset size**: on AMD, dot i8 goes from 55 ns
  (L2) to 174 ns (beyond L3) — 3.2x slower. On ARM, 25 ns to 69 ns — 2.8x slower.
- **Intel 130k barely worse than 2500** because 127 MB fits in the 480 MB L3 cache.
- **The random access penalty is where bulk operations with prefetching make a
  difference** — a library that can batch multiple vector comparisons and prefetch
  ahead can significantly reduce the cache miss cost.
