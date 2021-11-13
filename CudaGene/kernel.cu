#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "sharedData.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

extern "C" void
launch_addkernel(int size, int *c, const int *a, const int *b) {
    addKernel << <1, size >> > (c, a, b);
}

void unfoldAndUseDNA() {

}

__global__ void fitnessCalculation(Organism *organisms, int lengthX, FittingData fittingData, float *fitnessResults) {
    //fitness

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;


    fitnessResults[y*lengthX + x] =  y*lengthX + x; 

}

extern "C" void
launch_fitnessCalculation(int numOrganisms, Organism *organisms, FittingData fittingData, float *fitnessResults) {
    int numOrganismsX = (sqrt(numOrganisms)/8)*8;
    int numOrganismsY = numOrganisms / numOrganismsX;
    dim3 grid(numOrganismsX, numOrganismsY, 1);
    dim3 block(1, 1, 1);
    fitnessCalculation << < grid,block >> > (organisms, numOrganismsX, fittingData, fitnessResults);
}
