//
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaDP.cu
// ---
// 
//  This is the CUDA C implementation with Dynamic Parallelism. 
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <png.h>
#include <assert.h>

#include "cudaDP.h"
#include "common.h"
#include "defaults.h"
#include "metrics.h"

__host__ void cudaDPMandelbrotSets(int height, int width, int maxIterations, 
    const float radius, const complexNum cMin, const complexNum cMax, const char *filename) {
  const int OUTPUT_SIZE = height * width * sizeof(int);
  int *h_output = (int*) malloc(OUTPUT_SIZE);

  int *d_output = NULL; 
  cudaCheck(cudaMalloc((void **) &d_output, OUTPUT_SIZE));

  dim3 gridSize(MIN_SIZE, MIN_SIZE);
  dim3 blockSize(BLOCK_SIZE, DIVIDE_FACTOR);

  clock_t start = clock();

  cudaDPMandelbrotSetsKernel<<<gridSize, blockSize>>>(height, width, maxIterations,
      cMin, cMax, X_POS_DEFAULT, Y_POS_DEFAULT, width / MIN_SIZE, 1, radius,
      d_output);
  cudaCheck(cudaThreadSynchronize());
  
  endClock(start);

  if (filename != NULL) {
    cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));

    // Write output.
    saveImage(filename, h_output, width, height, maxIterations);
  }

 	// Free device output.
  cudaFree(d_output);
  free(h_output);
}

__device__ int commonValue(int v0, int v1, int maxIterations) {
  if (v0 == v1) {
    return v0;
  }
  if (v0 == maxIterations + 1 || v1 == maxIterations + 1) {
    return min(v0, v1);
  }
  return -1;
}

__device__ int calculateBorder(int width, int height, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, const float radius) {
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
  int blockSize = blockDim.x * blockDim.y;
  int value = maxIterations + 1;
  for (int pixel = tIdx; pixel < size; pixel += blockSize) {
    for (int boundary = 0; boundary < 4; boundary++) {
      int x = boundary % 2 != 0 ? x0 + pixel : (boundary == 0 ? x0 + size - 1 : x0); 
      int y = boundary % 2 == 0 ? y0 + pixel : (boundary == 1 ? y0 + size - 1 : y0);
      value = commonValue(value, calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius), maxIterations);
    }
  }

  __shared__ int s_output[BLOCK_SIZE * DIVIDE_FACTOR];
  int numThreads = min(size, BLOCK_SIZE * DIVIDE_FACTOR);
  if (tIdx < numThreads) {
    s_output[tIdx] = value;
  }
  __syncthreads();

  // while (numThreads > 1) {
  for(; numThreads > 1; numThreads /= 2) {
    if (tIdx < numThreads / 2) {
      s_output[tIdx] = commonValue(s_output[tIdx], s_output[tIdx + numThreads / 2], maxIterations);
    }
    __syncthreads();
  }
  return s_output[0];
}

__global__ void pixelKernel(int width, int height, int maxIterations, 
    complexNum cMin, complexNum cMax, int x0, int y0, int size, const float radius, int *d_output) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < size && y < size) {
    x += x0;
    y += y0;
    d_output[y * width + x] = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius);
  }
}

__global__ void fillKernel(int width, int x0, int y0, int size, int value, int *d_output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < size && y < size) {
    x += x0;
    y += y0;
    d_output[y * width + x] = value;
  }
}

__global__ void cudaDPMandelbrotSetsKernel(int height, int width, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int depth, const float radius,
    int *d_output) {

  x0 += size * blockIdx.x;
  y0 += size * blockIdx.y;

  int borderVal = calculateBorder(width, height, maxIterations, cMin, cMax, x0, y0, size, radius); 

  if(threadIdx.x == 0 && threadIdx.y == 0) {
    if (borderVal != -1) {
      dim3 fillBlockSize(BLOCK_SIZE, DIVIDE_FACTOR);
      dim3 fillGridSize(divup(size, BLOCK_SIZE), divup(size, DIVIDE_FACTOR));
      fillKernel<<<fillGridSize, fillBlockSize>>>(width, x0, y0, size, borderVal, d_output);
    }
    else if (depth + 1 < MAX_DEPTH && size / DIVIDE_FACTOR > MIN_SIZE) {
      dim3 recurseGridSize(DIVIDE_FACTOR, DIVIDE_FACTOR);
      dim3 recurseBlockSize(blockDim.x, blockDim.y);
      cudaDPMandelbrotSetsKernel<<<recurseGridSize, recurseBlockSize>>>(height, width, maxIterations, 
          cMin, cMax, x0, y0, size / DIVIDE_FACTOR, depth + 1, radius, d_output); 
    }
    else {
      dim3 pixelGridSize(divup(size, BLOCK_SIZE), divup(size, DIVIDE_FACTOR));
      dim3 pixelBlockSize(BLOCK_SIZE, DIVIDE_FACTOR);
      pixelKernel<<<pixelGridSize, pixelBlockSize>>>(width, height, maxIterations,
           cMin, cMax, x0, y0, size, radius, d_output);
    }
  }
}

