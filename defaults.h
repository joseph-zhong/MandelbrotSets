// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599 Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// defaults.h
// ---

#ifndef DEFAULTS_H
#define DEFAULTS_H
#include <stdbool.h>

#define VERBOSE true
#define KERNEL_VERBOSE true

// Colors.
#define CHANNELS 3

// Default values for specifying size, iterations, and resolution.
#define WIDTH_DEFAULT 600
#define HEIGHT_DEFAULT 400
#define MAX_ITER_DEFAULT 1000
#define X_POS_DEFAULT -0.5
#define Y_POS_DEFAULT 0.0
#define ZOOM_DEFAULT 1.0
#define RADIUS_DEFAULT 4.0

// CUDA Tile Sizes.
#define TILE_WIDTH 16

// Input kernels.
#define NAIVE_HOST "naive"
#define CUDA_NAIVE "cudaNaive"
#define CUDA_DP "cudaDP"

// Default output values.
#define OUT_DEFAULT "images/out.png"
#define NAIVE_OUT_DEFAULT "images/naive.png"
#define CUDA_NAIVE_OUT_DEFAULT "images/cudaNaive.png"
#define CUDA_DP_OUT_DEFAULT "images/cudaDP.png"

// Usage and example.
#define USAGE "Usage:   \n\t%s <width> <height> <maxiter> <kernel> <output> \n"
#define EXAMPLE "Example: \n\t%s 600 400 1000 naive images/out.png\n\n" 

// This is a debugging print statement which 
// prints if VERBOSE is set to true.
#define VERBOSE_PRINTF(str)                     \
{                                               \
  if(VERBOSE) {                                 \
    printf("\n");                               \
    printf(str);                                \
    printf("\n");                               \
  }                                             \
}         


#endif // DEFAULTS_H

