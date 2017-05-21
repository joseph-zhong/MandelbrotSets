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

__global__ void naiveMandelbrotSetsKernel(int height, int width, int maxIterations, 
     const float zoom, const float yPos, const float xPos, const float radius, 
     char *d_output, long long int *d_operations);

#endif // CUDA_NAIVE_H
