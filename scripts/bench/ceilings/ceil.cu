// Ceilings for the kernel optimisation plan: device copy bandwidth (read+write) and FP32 FMA peak.
#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

__global__ void copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void copy1(const float* __restrict__ a, float* __restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void readsum(const float4* __restrict__ a, float* out, size_t n) {
    float s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; s += v.x + v.y + v.z + v.w; }
    if (s == 1234.5f) out[0] = s;
}
__global__ void fma(float* out, int iters) {
    float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float m = 0.9999f, c = 0.0001f;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
            a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
        }
    }
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s == 1234.5f) out[0] = s;
}

template <typename F> float timeit(F f, int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    std::vector<float> t;
    for (int r = 0; r < reps; r++) { cudaEventRecord(s); f(); cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms, s, e); t.push_back(ms); }
    std::sort(t.begin(), t.end()); return t[t.size() / 2];
}

int main() {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    size_t bytes = 2ull << 30;  // 2 GiB per buffer
    float *a, *b, *o; cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMalloc(&o, 64);
    cudaMemset(a, 0, bytes);
    size_t n4 = bytes / 16, n1 = bytes / 4;
    int grid = sms * 8, block = 256;
    for (int w = 0; w < 5; w++) copy4<<<grid, block>>>((float4*)a, (float4*)b, n4);
    float ms4 = timeit([&] { copy4<<<grid, block>>>((float4*)a, (float4*)b, n4); }, 30);
    float ms1 = timeit([&] { copy1<<<grid, block>>>(a, b, n1); }, 30);
    float msr = timeit([&] { readsum<<<grid, block>>>((float4*)a, o, n4); }, 30);
    printf("copy float4: %.1f GB/s (read+write)\n", 2.0 * bytes / ms4 / 1e6);
    printf("copy float : %.1f GB/s (read+write)\n", 2.0 * bytes / ms1 / 1e6);
    printf("read float4: %.1f GB/s\n", 1.0 * bytes / msr / 1e6);
    int iters = 4096, fgrid = sms * 16, fblock = 256;
    fma<<<fgrid, fblock>>>(o, 64);
    float msf = timeit([&] { fma<<<fgrid, fblock>>>(o, iters); }, 15);
    double flops = 2.0 * 8 * 16 * (double)iters * fgrid * fblock;
    printf("fp32 fma   : %.2f TFLOP/s\n", flops / msf / 1e9);
    cudaError_t err = cudaGetLastError();
    printf("status: %s, SMs %d\n", cudaGetErrorString(err), sms);
    return 0;
}
