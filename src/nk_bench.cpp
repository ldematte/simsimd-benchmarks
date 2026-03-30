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

// ---------------------------------------------------------------------------
// Multi-vector benchmarks: loop over a dataset scoring many vectors.
// Equivalent to ES VectorScorerInt8BulkOperationBenchmark but without
// a native bulk operation — just single calls in a loop.
// ---------------------------------------------------------------------------

struct DatasetFixture {
    int dims;
    int num_vectors;
    int num_vectors_to_score;
    std::vector<int8_t> dataset;     // contiguous: num_vectors * dims
    std::vector<int8_t> query;       // dims
    std::vector<int> sequential_ids; // [0, 1, 2, ...]
    std::vector<int> random_ids;     // shuffled subset

    DatasetFixture(int dims, int num_vectors)
        : dims(dims), num_vectors(num_vectors),
          num_vectors_to_score(std::min(num_vectors, 20000)),
          dataset(static_cast<size_t>(num_vectors) * dims),
          query(dims) {

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-128, 127);

        for (auto &x : dataset) x = static_cast<int8_t>(dist(rng));
        for (auto &x : query) x = static_cast<int8_t>(dist(rng));

        // Sequential ids
        sequential_ids.resize(num_vectors);
        std::iota(sequential_ids.begin(), sequential_ids.end(), 0);

        // Random ordinals: shuffled then truncated to num_vectors_to_score
        random_ids = sequential_ids;
        std::shuffle(random_ids.begin(), random_ids.end(), rng);
        random_ids.resize(num_vectors_to_score);
    }

    const int8_t *vec_at(int ordinal) const {
        return dataset.data() + static_cast<size_t>(ordinal) * dims;
    }
};

// Args: {dims, numVectors}
static void MultiArgs(benchmark::internal::Benchmark *b) {
    // With dims=1024, each vector is 1KB:
    // 128 vectors = 128KB: overflows L1, fits in L2
    // 2500 vectors = 2.5MB: overflows L2 on AMD (1MB) and Graviton (2MB), fits in L3
    // 130000 vectors = ~127MB: overflows L3 on both
    for (int nv : {128, 2500, 130000})
        b->Args({1024, nv});
}

// Template for multi-vector i8 benchmarks that loop calling a single NK function.
// FuncType: void(*)(const int8_t*, const int8_t*, nk_size_t, ResultT*)
template <typename ResultT, typename FuncT>
static void BM_multi_i8(benchmark::State &state, FuncT func, bool random_access) {
    const int dims = static_cast<int>(state.range(0));
    const int num_vectors = static_cast<int>(state.range(1));

    static std::unordered_map<uint64_t, std::unique_ptr<DatasetFixture>> fixtures;
    uint64_t key = (static_cast<uint64_t>(dims) << 32) | static_cast<uint64_t>(num_vectors);
    if (fixtures.find(key) == fixtures.end()) {
        fixtures[key] = std::make_unique<DatasetFixture>(dims, num_vectors);
    }
    const auto &ds = *fixtures[key];

    const auto &ids = random_access ? ds.random_ids : ds.sequential_ids;
    const int n = ds.num_vectors_to_score;

    ResultT result;
    for (auto _ : state) {
        for (int v = 0; v < n; v++) {
            func(ds.vec_at(ids[v]), ds.query.data(),
                 static_cast<nk_size_t>(dims), &result);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// --- Dot product i8 ---
static void BM_multi_dot_i8_sequential(benchmark::State &s) {
    BM_multi_i8<nk_i32_t>(s, nk_dot_i8, false);
}
static void BM_multi_dot_i8_random(benchmark::State &s) {
    BM_multi_i8<nk_i32_t>(s, nk_dot_i8, true);
}
BENCHMARK(BM_multi_dot_i8_sequential)->Apply(MultiArgs);
BENCHMARK(BM_multi_dot_i8_random)->Apply(MultiArgs);

// --- Squared euclidean i8 ---
static void BM_multi_sqeuclidean_i8_sequential(benchmark::State &s) {
    BM_multi_i8<nk_u32_t>(s, nk_sqeuclidean_i8, false);
}
static void BM_multi_sqeuclidean_i8_random(benchmark::State &s) {
    BM_multi_i8<nk_u32_t>(s, nk_sqeuclidean_i8, true);
}
BENCHMARK(BM_multi_sqeuclidean_i8_sequential)->Apply(MultiArgs);
BENCHMARK(BM_multi_sqeuclidean_i8_random)->Apply(MultiArgs);

// --- Cosine i8 ---
static void BM_multi_angular_i8_sequential(benchmark::State &s) {
    BM_multi_i8<nk_f32_t>(s, nk_angular_i8, false);
}
static void BM_multi_angular_i8_random(benchmark::State &s) {
    BM_multi_i8<nk_f32_t>(s, nk_angular_i8, true);
}
BENCHMARK(BM_multi_angular_i8_sequential)->Apply(MultiArgs);
BENCHMARK(BM_multi_angular_i8_random)->Apply(MultiArgs);
