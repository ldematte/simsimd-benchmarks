#include <numkong/numkong.h>
#include <cstdio>
#include <chrono>

int main() {
    nk_capability_t caps = nk_capabilities();
    printf("Capabilities: 0x%llx\n", (unsigned long long)caps);
    printf("  serial:   %d\n", (caps & nk_cap_serial_k) != 0);
    printf("  neon:     %d\n", (caps & nk_cap_neon_k) != 0);
    printf("  neonsdot: %d\n", (caps & nk_cap_neonsdot_k) != 0);
    printf("  haswell:  %d\n", (caps & nk_cap_haswell_k) != 0);
    printf("  skylake:  %d\n", (caps & nk_cap_skylake_k) != 0);
    printf("  icelake:  %d\n", (caps & nk_cap_icelake_k) != 0);

    nk_configure_thread(caps);

    // Quick timing test: f32 dot 1024
    const int dims = 1024;
    float a[dims], b[dims];
    for (int i = 0; i < dims; i++) { a[i] = 0.1f; b[i] = 0.2f; }

    nk_f64_t result_f64;
    // Warmup
    for (int i = 0; i < 1000; i++) nk_dot_f32(a, b, dims, &result_f64);

    // Time via dispatch
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) nk_dot_f32(a, b, dims, &result_f64);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_dispatch = std::chrono::duration<double, std::nano>(t1 - t0).count() / 100000;
    printf("\nnk_dot_f32 (dispatch): %.1f ns, result=%f\n", ns_dispatch, result_f64);

    // Time serial directly
    nk_f64_t result_serial;
    nk_dot_f32_serial(a, b, dims, &result_serial);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) nk_dot_f32_serial(a, b, dims, &result_serial);
    t1 = std::chrono::high_resolution_clock::now();
    double ns_serial = std::chrono::duration<double, std::nano>(t1 - t0).count() / 100000;
    printf("nk_dot_f32_serial:     %.1f ns, result=%f\n", ns_serial, result_serial);

    // Time i8
    nk_i8_t ai[dims], bi[dims];
    for (int i = 0; i < dims; i++) { ai[i] = 10; bi[i] = 20; }
    nk_i32_t result_i32;
    for (int i = 0; i < 1000; i++) nk_dot_i8(ai, bi, dims, &result_i32);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) nk_dot_i8(ai, bi, dims, &result_i32);
    t1 = std::chrono::high_resolution_clock::now();
    double ns_i8 = std::chrono::duration<double, std::nano>(t1 - t0).count() / 100000;
    printf("nk_dot_i8  (dispatch): %.1f ns, result=%d\n", ns_i8, result_i32);

    return 0;
}
