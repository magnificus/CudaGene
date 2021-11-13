#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "sharedData.h"

extern "C" void
launch_addkernel(int size, int* c, const int* a, const int* b);

extern "C" void
launch_fitnessCalculation(int numOrganisms, int *organisms, FittingData fittingData, int *fitnessResults);


#define INPUT_DATA_SIZE (sizeof(INPUT_DATA_TYPE) * DATA_LENGTH)
#define GROUND_TRUTH_SIZE (sizeof(OUTPUT_DATA_TYPE) * DATA_LENGTH)

#define ORGANISMS_SIZE (sizeof(Organism) * NUM_ORGANISMS)
#define FITNESS_RESULT_SIZE (sizeof(float) * NUM_ORGANISMS)

FittingData setupFittingData() {
    // this is where you define
    FittingData toReturn;
    toReturn.inputData = (INPUT_DATA_TYPE*) malloc(INPUT_DATA_SIZE);
    toReturn.groundTruth = (OUTPUT_DATA_TYPE*) malloc(GROUND_TRUTH_SIZE);

    for (unsigned int i = 0; i < DATA_LENGTH; i++) {
        toReturn.inputData[i] = i;
        toReturn.groundTruth[i] = i * 2;
    }
    return toReturn;
}

Organism* setupInitialDNA() {
    Organism* organisms = (Organism*)malloc(ORGANISMS_SIZE);
    for (unsigned int i = 0; i < NUM_ORGANISMS; i++) {
        for (unsigned int j = 0; j < DNA_LENGTH; j++) {
            organisms[i].DNA[j] = 1;
        }
    }
    return organisms;
}

void clearDNA(Organism *organisms) {
    free(organisms);
}

void clearFittingData(FittingData fittingData) {
    free(fittingData.groundTruth);
    free(fittingData.inputData);
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    FittingData fittingData = setupFittingData();
    FittingData fittingDataOnGPU;
    Organism* organisms = setupInitialDNA();
    Organism* organismsOnGPU;
    float *fitnessResults = (float*) malloc(FITNESS_RESULT_SIZE);
    float *fitnessResultsOnGPU;

    // GPU allocation
    cudaMalloc((void**)&fittingDataOnGPU.inputData, INPUT_DATA_SIZE);
    cudaMalloc((void**)&fittingDataOnGPU.groundTruth, INPUT_DATA_SIZE);
    cudaMalloc((void**)&organismsOnGPU, ORGANISMS_SIZE);
    cudaMalloc((void**)&fitnessResults, FITNESS_RESULT_SIZE);

    // copy from local memory to GPU


    cudaMemcpy(fittingDataOnGPU.inputData, fittingData.inputData, INPUT_DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(fittingDataOnGPU.groundTruth, fittingData.groundTruth, INPUT_DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(organismsOnGPU, organisms, ORGANISMS_SIZE, cudaMemcpyHostToDevice);


    /*// Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    */

    // Launch a kernel on the GPU with one thread for each element.
    //launch_addkernel(size, dev_c, dev_a, dev_b);

    //launch_fitnessCalculation(NUM_ORGANISMS, nullptr, nullptr, nullptr);

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
    //cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:

    clearFittingData(fittingData);
    clearDNA(organisms);
    //cudaFree(dev_c);
    //cudaFree(dev_a);
    //cudaFree(dev_b);

    return cudaStatus;
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
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
