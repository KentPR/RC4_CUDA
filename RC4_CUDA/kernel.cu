#define _CRT_SECURE_NO_WARNINGS

#define SRC_FILE_PATH "2048x2048.bmp"
#define DEST_FILE_PATH "result.bmp"
#define RESULT_TXT "result.txt"
#define TEXT_IN_PATH "tin.txt"
#define TEXT_OUT_PATH "tout.txt"
#define KEY_PATH "key.txt"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <malloc.h>
#include <math.h>
#include <stdio.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] * b[i];
//}

cudaError_t xorWithCuda(int* out, unsigned int* in, unsigned int* key, unsigned int XORlength);

__global__ void xorKernel(unsigned int* out, unsigned int* in, unsigned int* key)
{
    int i = threadIdx.x;
    out[i] = in[i] ^ key[i];
}

__global__ void dec2bin_arrayKernel(unsigned int* dec, unsigned int* bin, int l10)
{
    int i = threadIdx.x;
    if (i <= l10)
    {
        int dec_b = dec[i];
        for (int j = 0; j < 8; j++)
        {
            bin[i * 8 + j] = dec_b % 2;
            dec_b /= 2;
        }
    }
}

__global__ void bin2dec_arrayKernel(unsigned int* dec, unsigned int* bin, int l10)
{
    int i = threadIdx.x;
    if (i <= l10)
    {
        for (int j = 0; j < 8; j++)
        {
            int value = pow(2, j) * bin[i * 8 + j];
            dec[i] += value;
        }
    }
}

int main()
{
    /*const int arraySize = 5;
    unsigned int a[arraySize] = { 1, 2, 3, 4, 5 };
    unsigned int b[arraySize] = { 10, 20, 30, 40, 50 };
    unsigned int c[arraySize] = { 0 };*/

    

    FILE* KEYfile = fopen(KEY_PATH, "r");
    if (KEYfile == NULL)
    {
        printf("ERROR: unable to open file with message.\n");
        return 0;
    }
    fseek(KEYfile, 0L, SEEK_END);
    int KEYSize = ftell(KEYfile);
    fseek(KEYfile, 0L, SEEK_SET);
    int KEY2Size = KEYSize * 8;
    unsigned int* key10 = (unsigned int*)malloc(sizeof(unsigned int) * KEYSize);
    for (int i = 0; i < KEYSize; i++) key10[i] = fgetc(KEYfile);
    unsigned int* key2 = (unsigned int*)malloc(sizeof(unsigned int) * KEY2Size);

    int sw; //switch

    // Xor vectors in parallel.
    cudaError_t cudaStatus = xorWithCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} * {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t xorWithCuda(int* out, unsigned int* in, unsigned int* key, unsigned int XORlength)
{
    unsigned int* dev_in = 0;
    unsigned int* dev_key = 0;
    unsigned int* dev_out = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_out, XORlength * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_out cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, XORlength * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_in cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_key, XORlength * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_key cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, XORlength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_key, key, XORlength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    xorKernel << <1, size >> > (dev_out, dev_in, dev_key);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "xorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, XORlength * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_key);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel << <1, size >> > (dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
