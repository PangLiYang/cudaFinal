#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

using namespace std;

const int THREADS_PER_BLOCK = 32;
//const int N = 232960 >> 8 << 8;
//const int N = 716847 >> 8 << 8;
const int N = 2449029 >> 8 << 8;
//const int N = 4096;

const int dim_in = 256, dim_out = 64;

__global__ void maxpool(float *, float *, unsigned int *);

int main() {

    cout<<"N = "<< N << ", dim_in = " << dim_in << ", dim_out = " << dim_out << ", preparing data..." << endl;

    float *data, *value;
    unsigned int *indices;

    cudaMallocManaged(&data,  N * dim_in  * sizeof(float));
    cudaMallocManaged(&value, N * dim_out * sizeof(float));
    cudaMallocManaged(&indices, N * dim_out * sizeof(unsigned int));

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    generate(data, data + N * dim_in, [&](){ return rd(engine); });

    unsigned int shared_mem_size = THREADS_PER_BLOCK * dim_in * sizeof(float);

    cout<<"Config GridDim = "<< N / THREADS_PER_BLOCK << ", BlockDim = " << THREADS_PER_BLOCK << ", shared_mem_size = " << shared_mem_size << endl;

    dim3 grid(N / THREADS_PER_BLOCK, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    int times = 100;
    for (int i = 0; i < times; i++) {
        maxpool <<< grid, block, shared_mem_size >>> (data, value, indices);
    }

    cudaDeviceSynchronize();
    double measured_time = 0;

    for (int i = 0; i < times; i++) {
        timestamp(t0);
        maxpool <<< grid, block, shared_mem_size >>> (data, value, indices);
        cudaDeviceSynchronize();
        timestamp(t1);
        measured_time += getDuration(t0, t1);
    }

    cout << "max-pooling time = " << measured_time / times * 1000 << " ms" <<endl;

    cudaDeviceSynchronize();

    for (int i = 0; i < 64; i += 1) {
        cout << "value[" << i << "] = " << value[i] << endl;
    }

    for (int i = 0; i < 64; i += 1) {
        cout << "indices[" << i << "] = " << indices[i] << endl;
    }

    cudaFree(data);
    cudaFree(value);
    cudaFree(indices);

    return 0;
}

__global__ void maxpool(float *data, float *value, unsigned int *indices) {

    extern __shared__ float buffer[];

    const int sqrt_dim_in = 16;
    const int thread_offset = threadIdx.x * dim_in;
    const int block_offset = blockIdx.x * THREADS_PER_BLOCK * dim_in;

#pragma unroll
    for (unsigned int i = 0; i < dim_in; i += 1) {
        buffer[thread_offset + i] = data[block_offset + thread_offset + i];
    }

    //__syncwarp();

    float v;
    int pos;
    int offset = 0;

#pragma unroll
    for (int xx = 0; xx < sqrt_dim_in; xx += 2) {

        for (int yy = 0; yy < sqrt_dim_in; yy += 2) {

            pos = xx * sqrt_dim_in + yy;
            v = buffer[thread_offset + pos];

            if (buffer[thread_offset + xx * sqrt_dim_in + yy + 1] > v) {
                pos = xx * sqrt_dim_in + yy + 1;
                v = buffer[thread_offset + pos];
            }

            if (buffer[thread_offset + (xx + 1) * sqrt_dim_in + yy] > v) {
                pos = (xx + 1) * sqrt_dim_in + yy;
                v = buffer[thread_offset + pos];
            }

            if (buffer[thread_offset + (xx + 1) * sqrt_dim_in + yy + 1] > v) {
                pos = (xx + 1) * sqrt_dim_in + yy + 1;
                v = buffer[thread_offset + pos];
            }

            value[blockIdx.x * THREADS_PER_BLOCK * dim_out + threadIdx.x * dim_out + offset] = v;
            indices[blockIdx.x * THREADS_PER_BLOCK * dim_out + threadIdx.x * dim_out + offset] = pos;

            offset += 1;

        }
    }
}