// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaNaive.h
// ---
// 
//  This is the naive CUDA C implementation without usage of Dynamic
//  Parallelism. 
//

#ifndef CUDA_NAIVE_H
#define CUDA_NAIVE_H

#include "cudaCommon.h"

__host__ void cudaNaiveMandelbrotSets(int height, int width, int maxIterations, 
        const float radius, const char *filename);

// __global__ void cudaNaiveMandelbrotSetsKernel(int height, int width, int maxIterations, 
//      const float zoom, const float yPos, const float xPos, const float radius, 
//      int *d_output, long long int *d_operations);

__global__ void cudaNaiveMandelbrotSetsKernel(int *d_output, 
  int width, int height, int maxIterations, const float radius, 
  complexNum cmin, complexNum cmax);

#endif // CUDA_NAIVE_H

