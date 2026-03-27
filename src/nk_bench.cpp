#include <benchmark/benchmark.h>
#include <numkong/numkong.h>

#include <cstdint>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Random data generation (seeded for reproducibility)
// ---------------------------------------------------------------------------

static std::vector<float> random_f32_vector(int dims) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dims);
    for (auto &x : v) x = dist(rng);
    return v;
}

static std::vector<int8_t> random_i8_vector(int dims) {
    static std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> v(dims);
    for (auto &x : v) x = static_cast<int8_t>(dist(rng));
    return v;
}

// ---------------------------------------------------------------------------
// Dimensions: 128, 256, 384, 512, 768, 1024, 1536, 3072
// ---------------------------------------------------------------------------

static void DimRange(benchmark::internal::Benchmark *b) {
    for (int d : {128, 256, 384, 512, 768, 1024, 1536, 3072})
        b->Arg(d);
}

// ---------------------------------------------------------------------------
// One-time setup: configure NumKong's dynamic dispatch for this thread.
// This selects the best available ISA (e.g. AVX2, AVX-512, NEON+SDOT)
// based on runtime CPU detection — same as production usage.
// ---------------------------------------------------------------------------

static bool nk_configured = [] {
    nk_configure_thread(nk_capabilities());
    return true;
}();

// ---------------------------------------------------------------------------
// Benchmark macros
// ---------------------------------------------------------------------------

#define BENCH_F32(Name, Func)                                          \
    static void Name(benchmark::State &state) {                        \
        const int dims = static_cast<int>(state.range(0));             \
        auto a = random_f32_vector(dims);                              \
        auto b = random_f32_vector(dims);                              \
        nk_f64_t result;                                               \
        for (auto _ : state) {                                         \
            Func(a.data(), b.data(), static_cast<nk_size_t>(dims),     \
                 &result);                                             \
            benchmark::DoNotOptimize(result);                          \
        }                                                              \
        state.SetItemsProcessed(state.iterations());                   \
    }                                                                  \
    BENCHMARK(Name)->Apply(DimRange)

#define BENCH_I8_DOT(Name, Func)                                       \
    static void Name(benchmark::State &state) {                        \
        const int dims = static_cast<int>(state.range(0));             \
        auto a = random_i8_vector(dims);                               \
        auto b = random_i8_vector(dims);                               \
        nk_i32_t result;                                               \
        for (auto _ : state) {                                         \
            Func(a.data(), b.data(), static_cast<nk_size_t>(dims),     \
                 &result);                                             \
            benchmark::DoNotOptimize(result);                          \
        }                                                              \
        state.SetItemsProcessed(state.iterations());                   \
    }                                                                  \
    BENCHMARK(Name)->Apply(DimRange)

#define BENCH_I8_SQE(Name, Func)                                       \
    static void Name(benchmark::State &state) {                        \
        const int dims = static_cast<int>(state.range(0));             \
        auto a = random_i8_vector(dims);                               \
        auto b = random_i8_vector(dims);                               \
        nk_u32_t result;                                               \
        for (auto _ : state) {                                         \
            Func(a.data(), b.data(), static_cast<nk_size_t>(dims),     \
                 &result);                                             \
            benchmark::DoNotOptimize(result);                          \
        }                                                              \
        state.SetItemsProcessed(state.iterations());                   \
    }                                                                  \
    BENCHMARK(Name)->Apply(DimRange)

#define BENCH_I8_ANG(Name, Func)                                       \
    static void Name(benchmark::State &state) {                        \
        const int dims = static_cast<int>(state.range(0));             \
        auto a = random_i8_vector(dims);                               \
        auto b = random_i8_vector(dims);                               \
        nk_f32_t result;                                               \
        for (auto _ : state) {                                         \
            Func(a.data(), b.data(), static_cast<nk_size_t>(dims),     \
                 &result);                                             \
            benchmark::DoNotOptimize(result);                          \
        }                                                              \
        state.SetItemsProcessed(state.iterations());                   \
    }                                                                  \
    BENCHMARK(Name)->Apply(DimRange)

// ---------------------------------------------------------------------------
// Benchmarks — use dynamically-dispatched functions (via shared library).
// At runtime, nk_configure_thread selects the best ISA for this CPU.
// ---------------------------------------------------------------------------

// f32: dot product and squared euclidean only (cosine f32 skipped — with
// normalized vectors dot product and cosine are equivalent)
BENCH_F32(BM_dot_f32,          nk_dot_f32);
BENCH_F32(BM_sqeuclidean_f32,  nk_sqeuclidean_f32);

// i8: dot product, squared euclidean, and cosine
BENCH_I8_DOT(BM_dot_i8,        nk_dot_i8);
BENCH_I8_SQE(BM_sqeuclidean_i8, nk_sqeuclidean_i8);
BENCH_I8_ANG(BM_angular_i8,    nk_angular_i8);
