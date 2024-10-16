#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cuda_runtime.h>

using namespace std::chrono;

void matmulCPU(int* v1, int* v2, int* out, size_t N){
    for(int i = 0; i < N; i++){
        out[i] = v1[i] * v2[i];
    }
}

__global__ void matmul(int* v1, int* v2, int* out, size_t N){
    // printf("blockIdx: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        out[i] = v1[i] * v2[i];
    }
}

int main(void){
    int N = 1'000'000'000;
    size_t SIZE = N * sizeof(int);

    int *h_vec1, *h_vec2, *h_out;
    h_vec1 = (int*)calloc(N, sizeof(int));
    h_vec2 = (int*)calloc(N, sizeof(int));
    h_out = (int*)calloc(N, sizeof(int));
    if(h_vec1 == nullptr || h_vec2 == nullptr || h_out == nullptr){
        printf("Malloc failed for host arrays\n");
        return 1;
    }

    srand(42);
    for(int i = 0; i < N; i++){
        h_vec1[i] = (rand() % 255) + 1;
        h_vec2[i] = (rand() % 255) + 1;
    }

    int* d_v1;
    int* d_v2;
    int* d_out;

    std::cout << "Start GPU" << std::endl;
    auto start_cuda = high_resolution_clock::now();
    cudaMalloc((void **)&d_v1, SIZE);
    cudaMalloc((void **)&d_v2, SIZE);
    cudaMalloc((void **)&d_out, SIZE);

    cudaMemcpy(d_v1,h_vec1, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2,h_vec2, SIZE, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_v1, d_v2, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost);
    auto end_cuda = high_resolution_clock::now();
    auto cuda_time = duration_cast<microseconds>(end_cuda - start_cuda);
    std::cout << "End GPU" << std::endl;
    std::cout << "Time: " << cuda_time.count() << std::endl;

    std::cout << "Start CPU" << std::endl;
    auto start_cpu = high_resolution_clock::now();
    matmulCPU(h_vec1, h_vec2, h_out, N);
    auto end_cpu = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(end_cpu - start_cpu);
    std::cout << "End CPU" << std::endl;
    std::cout << "Time: " << cpu_time.count() << std::endl;

    free(h_out);
    free(h_vec1);
    free(h_vec2);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_out);
}