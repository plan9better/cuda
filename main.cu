#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 2; // 2x2 matrices
    const int SIZE = N * N * sizeof(float);

    // Host matrices
    float h_A[N*N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[N*N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[N*N] = {0.0f};

    // Device matrices
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, SIZE);
    cudaMalloc((void**)&d_B, SIZE);
    cudaMalloc((void**)&d_C, SIZE);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    // Launch kernel

    // Synchronize to make sure the kernel has finished
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_C[i*N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
