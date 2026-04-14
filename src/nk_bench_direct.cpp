#include <benchmark/benchmark.h>
#include <numkong/numkong.h>

// Include ISA-specific headers directly for direct calls
#if defined(__x86_64__) || defined(_M_X64)
#include <numkong/dot/haswell.h>
#include <numkong/dot/icelake.h>
#include <numkong/dot/skylake.h>
#include <numkong/spatial/haswell.h>
#include <numkong/spatial/icelake.h>
#include <numkong/spatial/skylake.h>
#define HAS_X86 1
#endif

#include <cstdint>
#include <random>
#include <vector>

static std::vector<float> random_f32_vector(int dims, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dims);
    for (auto &x : v) x = dist(rng);
    return v;
}

static std::vector<int8_t> random_i8_vector(int dims, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> v(dims);
    for (auto &x : v) x = static_cast<int8_t>(dist(rng));
    return v;
}

static void DimRange(benchmark::internal::Benchmark *b) {
    for (int d : {128, 256, 512, 1024, 3072})
        b->Arg(d);
}

// --- Dynamic dispatch (what our benchmarks used) ---

static bool nk_configured = [] {
    nk_configure_thread(nk_capabilities());
    return true;
}();

static void BM_dot_i8_dispatch(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_i32_t result;
    for (auto _ : state) {
        nk_dot_i8(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_i8_dispatch)->Apply(DimRange);

static void BM_sqe_i8_dispatch(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_u32_t result;
    for (auto _ : state) {
        nk_sqeuclidean_i8(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_i8_dispatch)->Apply(DimRange);

#if HAS_X86
// --- Direct Ice Lake calls (bypass dispatch) ---

static void BM_dot_i8_icelake(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_i32_t result;
    for (auto _ : state) {
        nk_dot_i8_icelake(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_i8_icelake)->Apply(DimRange);

static void BM_sqe_i8_icelake(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_u32_t result;
    for (auto _ : state) {
        nk_sqeuclidean_i8_icelake(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_i8_icelake)->Apply(DimRange);

// --- Direct Haswell calls (for comparison) ---

static void BM_dot_i8_haswell(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_i32_t result;
    for (auto _ : state) {
        nk_dot_i8_haswell(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_i8_haswell)->Apply(DimRange);

static void BM_sqe_i8_haswell(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_i8_vector(dims, 42);
    auto b = random_i8_vector(dims, 123);
    nk_u32_t result;
    for (auto _ : state) {
        nk_sqeuclidean_i8_haswell(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_i8_haswell)->Apply(DimRange);

// --- f32: dispatch (static inline, resolves to skylake with -march=icelake-client) ---

static void BM_dot_f32_dispatch(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_dot_f32(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_f32_dispatch)->Apply(DimRange);

static void BM_sqe_f32_dispatch(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_sqeuclidean_f32(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_f32_dispatch)->Apply(DimRange);

// --- f32: direct Skylake (AVX-512, f64 accumulator) ---

static void BM_dot_f32_skylake(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_dot_f32_skylake(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_f32_skylake)->Apply(DimRange);

static void BM_sqe_f32_skylake(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_sqeuclidean_f32_skylake(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_f32_skylake)->Apply(DimRange);

// --- f32: direct Haswell (AVX2, f64 accumulator) ---

static void BM_dot_f32_haswell(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_dot_f32_haswell(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_dot_f32_haswell)->Apply(DimRange);

static void BM_sqe_f32_haswell(benchmark::State &state) {
    const int dims = static_cast<int>(state.range(0));
    auto a = random_f32_vector(dims, 42);
    auto b = random_f32_vector(dims, 123);
    nk_f64_t result;
    for (auto _ : state) {
        nk_sqeuclidean_f32_haswell(a.data(), b.data(), static_cast<nk_size_t>(dims), &result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_sqe_f32_haswell)->Apply(DimRange);
#endif
