#define _CRT_SECURE_NO_WARNINGS

#define SRC_FILE_PATH "2048x2048.bmp"
#define DEST_FILE_PATH "result.bmp"
#define RESULT_TXT "result.txt"
#define TEXT_IN_PATH "tin.txt"
#define TEXT_OUT_PATH "tout.txt"
#define KEY_PATH "key.txt"

#define MAX_TH 1024
#define BYTE 8

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <malloc.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

int blocks;
int threadsperblock;
float TiMe = 0.0;



__host__ void retribution(void)
{
    blocks = 0;
    threadsperblock = 0;
}
__host__ int distribution(int N)
{
    if (N < 1)
        return -1;
    if (N < MAX_TH)
    {
        blocks = 1;
        for (int i = 0; i < 10; i++)
        {
            threadsperblock = (int)pow(2, i);
            if (threadsperblock >= N)
            {
                break;
            }
        }
    }
    else
    {
        threadsperblock = MAX_TH;
        if (N % MAX_TH == 0)
            blocks = N / MAX_TH;
        else
            blocks = (N / MAX_TH) + 1;
    }
    //printf("Kernel will be distributed with %d blocks, %d threads per block\n", blocks, threadsperblock);
    return 0;
}

cudaError_t xorWithCuda(unsigned int* c, unsigned int* a, unsigned int* b, int* count);
cudaError_t dec2bin_arrayWithCuda(unsigned int* dec, unsigned int* bin, int* len10, int* len2);
cudaError_t bin2dec_arrayWithCuda(unsigned int* bin, unsigned int* dec, int* len10, int* len2);

__global__ void dec2bin_arrayKernel(unsigned int* bin, unsigned int* dec, int count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //indexing in parallel operations
    if (tid <= count)
    {
        int dec_b = dec[tid]; // temporary varible
        for (int j = 0; j < BYTE; j++)
        {
            bin[tid * BYTE + j] = dec_b % 2;
            dec_b /= 2;
        }
    }
}

__global__ void bin2dec_arrayKernel(unsigned int* dec, unsigned int* bin, int count, const unsigned int* SUP)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //indexing in parallel operations
    if (tid <= count)
    {
        dec[tid] = 0;
        for (int j = 0; j < BYTE; j++)
        {
            int value = SUP[j] * bin[tid * BYTE + j];
            dec[tid] += value;
        }
    }
}

__global__ void xorKernel(unsigned int* c, unsigned int* a, unsigned int* b, int count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= count)
        c[tid] = a[tid] ^ b[tid];
}

__host__ void swap(unsigned int* i, unsigned int* j);
__host__ void init_Sblock(unsigned int* S, unsigned int* key, int L);
__host__ int copyFile(FILE* ifp, FILE* ofp, unsigned int* in10);
__host__ void PRGA(unsigned int* S, int len, unsigned int* ext);
__host__ void RC4encryption(unsigned int* key, unsigned int* in, unsigned int* out, int pic2size, int keylen);

int main()
{
    cudaError_t cudaStatus;

    FILE* KEYfile = fopen(KEY_PATH, "r");
    if (KEYfile == NULL)
    {
        printf("ERROR: unable to open file with message.\n");
        return 0;
    }
    fseek(KEYfile, 0L, SEEK_END);
    int KEYSize = ftell(KEYfile); //getting the number of symbols/bytes in KEYfile
    fseek(KEYfile, 0L, SEEK_SET);
    int KEY2Size = KEYSize * 8;
    unsigned int* key10 = new unsigned int[KEYSize]; //memory allocation for key10
    for (int i = 0; i < KEYSize; i++) key10[i] = fgetc(KEYfile); //copy information from KEYfile to key10 array
    unsigned int* key2 = new unsigned int[KEY2Size]; //memory allocation for key2


    //Open Image file
    FILE* imgFile = fopen(SRC_FILE_PATH, "rb");
    if (imgFile == NULL)
    {
        printf("ERROR: unable to open source file.\n");
        return 0;
    }

    fseek(imgFile, 0L, SEEK_END);
    int picSize = ftell(imgFile) - 54; // number of bytes in image
    int pic2Size = picSize * 8;   // number of bits in image

    unsigned int* in10 = (unsigned int*)malloc(picSize * sizeof(unsigned int));
    unsigned int* out10 = (unsigned int*)malloc(picSize * sizeof(unsigned int));
    unsigned int* in2 = (unsigned int*)malloc(pic2Size * sizeof(unsigned int));
    unsigned int* out2 = (unsigned int*)malloc(pic2Size * sizeof(unsigned int));
    for (int i = 0; i < pic2Size; i++) in2[i] = 0;
    for (int i = 0; i < picSize; i++) out10[i] = 0;

    //Создать копию файла - изображения
    FILE* destFile = fopen(DEST_FILE_PATH, "wb+");
    if (destFile == NULL)
    {
        printf("ERROR: unable to create destination file.\n");
        return 0;
    }
    fseek(imgFile, 0L, SEEK_SET);
    fseek(destFile, 0L, SEEK_SET);
    /*copy file
    &
    filling in10[] array beginning from 54th byte*/
    copyFile(imgFile, destFile, in10);


    cudaStatus = dec2bin_arrayWithCuda(in10, in2, &picSize, &pic2Size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dec2bin_arrayWithCuda failed!");
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    RC4encryption(key10, in2, out2, pic2Size, KEYSize);

    cudaStatus = bin2dec_arrayWithCuda(out2,out10, &picSize, &pic2Size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "bin2dec_arrayWithCuda failed!");
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    fseek(destFile, 0x36, SEEK_SET);
    for (int i = 0; i < picSize; i++)
    {
        char ch = (char)out10[i];
        fputc(ch, destFile);
    }
    fclose(destFile);

    printf("Total time = %0.2f miliseconds\n", TiMe);

    return 0;
}

__host__ void swap(unsigned int* i, unsigned int* j)
{
    unsigned int temp;

    temp = *i;
    *i = *j;
    *j = temp;
    return;
}

__host__ void init_Sblock(unsigned int* S, unsigned int* key, int L)
{
    for (int i = 0; i < 256; i++)
        S[i] = i;
    int j = 0;
    for (int i = 0; i < 256; i++)
    {
        j = (j + S[i] + key[i % L]) % 256;
        swap(&S[i], &S[j]);
    }
}

__host__ int copyFile(FILE* ifp, FILE* ofp, unsigned int* in10)
{
    int chread;
    int counter = 0;
    while ((chread = fgetc(ifp)) != EOF)
    {
        char ch = chread;
        if (counter >= 54) in10[counter - 54] = chread;
        putc(chread, ofp);
        counter++;
    }
    return counter;
}

__host__ void PRGA(unsigned int* S, int len, unsigned int* ext)
{
    int i = 0; int j = 0;
    for (int k = 0; k < len; k++)
    {
        i = (i + 1) % 256;
        j = (j + S[i]) % 256;
        swap(&S[i], &S[j]);
        ext[k] = S[(S[i] + S[j]) % 256];
    }
}

__host__ void RC4encryption(unsigned int* key, unsigned int* in, unsigned int* out, int pic2size, int keylen)
{
    cudaError_t cudaStatus;
    int length10 = pic2size / 8;
    unsigned int S[256]; //S-block
    //Must have got the same length that an input/output arrays
    unsigned int* extended10_KEY = (unsigned int*)malloc(sizeof(unsigned int*) * length10);
    unsigned int* extended2_KEY = (unsigned int*)malloc(sizeof(unsigned int*) * pic2size);

    init_Sblock(S, key, keylen);
    PRGA(S, length10, extended10_KEY);

    cudaStatus = dec2bin_arrayWithCuda(extended10_KEY, extended2_KEY, &length10, &pic2size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dec2bin_arrayWithCuda failed!");
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    free(extended10_KEY);

    // Xor vectors in parallel.
    cudaStatus = xorWithCuda(out, in, extended2_KEY, &pic2size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "xorWithCuda failed!");
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    free(extended2_KEY);
}

// Helper function for using CUDA to convert decimal vectors to binary vectors in parallel.
cudaError_t dec2bin_arrayWithCuda(unsigned int* dec, unsigned int* bin, int* len10, int* len2)
{
    unsigned int* dev_dec = 0;
    unsigned int* dev_bin = 0;
    int* size10 = 0;
    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    float gpuTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_bin, *len2 * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_dec, *len10 * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&size10, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_dec, dec, *len10 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy size of input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(size10, &len10, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int distrSuccess = distribution(*len10);
    if (distrSuccess != 0) {
        printf("Unable to distribute grid and blocks!");
    }

    // Launch a kernel on the GPU with distributed parameters.
    dec2bin_arrayKernel << <blocks, threadsperblock >> > (dev_bin, dev_dec, *len10);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("dec2bin time on GPU = %f miliseconds\n", gpuTime);
    TiMe += gpuTime;

    retribution();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(bin, dev_bin, (*len2) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_bin);
    cudaFree(dev_dec);
    cudaFree(size10);

    return cudaStatus;
}

// Helper function for using CUDA to convert binary vectors to decimal vectors in parallel.
cudaError_t bin2dec_arrayWithCuda(unsigned int* bin, unsigned int* dec, int* len10, int* len2)
{
    unsigned int* dev_bin = 0;
    unsigned int* dev_dec = 0;
    int* size10 = 0;
    unsigned int* dev_sup = 0;
    cudaError_t cudaStatus;
    unsigned int sup[BYTE] = { 1, 2, 4, 8, 16, 32, 64, 128 };

    cudaEvent_t start, stop;
    float gpuTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_dec, *len10 * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_bin, *len2 * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&size10, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_sup, BYTE * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_bin, bin, *len2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy size of input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(size10, &len10, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy suport vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_sup, sup, BYTE * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int distrSuccess = distribution(*len10);
    if (distrSuccess != 0) {
        printf("Unable to distribute grid and blocks!");
    }

    // Launch a kernel on the GPU with distributed parameters.
    bin2dec_arrayKernel << <blocks, threadsperblock >> > (dev_dec, dev_bin, *len10, dev_sup);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("bin2dec time on GPU = %f miliseconds\n", gpuTime);
    TiMe += gpuTime;

    retribution();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bin2dec_arrayKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dec, dev_dec, (*len10) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_dec);
    cudaFree(dev_bin);
    cudaFree(size10);

    return cudaStatus;
}

// Helper function for using CUDA to vor vectors in parallel.
cudaError_t xorWithCuda(unsigned int* out, unsigned int* in, unsigned int* key, int* count)
{
    unsigned int* dev_in = 0;
    unsigned int* dev_key = 0;
    unsigned int* dev_out = 0;
    int* size = 0;
    cudaError_t cudaStatus;

    cudaEvent_t start, stop;
    float gpuTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_out, *count * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, *count * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_key, *count * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&size, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, *count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_key, key, *count * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(size, &count, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int distrSuccess = distribution(*count);
    if (distrSuccess != 0) {
        printf("Unable to distribute grid and blocks!");
    }

    // Launch a kernel on the GPU with one thread for each element.
    xorKernel << <blocks, threadsperblock >> > (dev_out, dev_in, dev_key, *count);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("xor time on GPU = %f miliseconds\n", gpuTime);
    TiMe += gpuTime;

    retribution();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(out, dev_out, (*count) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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