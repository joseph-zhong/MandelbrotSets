// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaCommon.cu
// ---
//
//  Common CUDA C code.
//

#include "cudaCommon.h"

// This computes ceil(x / y).
__host__ __device__ int divup(int x, int y);

__device__ int calculatePixelValue(int width, int height, int maxIterations,
    complexNum cMin, complexNum cMax, int x, int y, 
    const float radius) {
  
  complexNum diff = cMax - cMin;

  float fx = (float) x / width;
  float fy = (float) y / height;

  complexNum c = cMin + complexNum(fx * diff.a, fy * diff.bi);

  int iterations = 0;
  complexNum z = c;
  while (iterations < maxIterations && absSquared(z) < radius) {
    z = z * z + c;
    iterations++;
  }
  return iterations;
}

__host__ __device__ int divup(int x, int y) { 
  return x / y + (x % y ? 1 : 0); 
}
