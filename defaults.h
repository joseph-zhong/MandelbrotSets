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
#define KERNEL_VERBOSE false

// Colors.
#define CHANNELS 3

// Default values for specifying size, iterations, and resolution.
#define WIDTH_DEFAULT 600
#define HEIGHT_DEFAULT 400
#define MAX_ITER_DEFAULT 1000
#define RADIUS_DEFAULT 4.0

// Default location values.
#define X_MIN_DEFAULT -1.5
#define X_MAX_DEFAULT 0.5
#define Y_MIN_DEFAULT -1.0
#define Y_MAX_DEFAULT 1.0

// CUDA Tile Sizes.
#define TILE_WIDTH 16

// Default Dynamic Parallelism parameters.
#define BLOCK_SIZE 64
#define DIVIDE_FACTOR 4
#define MAX_DEPTH 4
#define MIN_SIZE 32

// Experiment directory.
#define EXPERIMENT_DIR "experiments"

// Input kernels.
#define NAIVE_HOST "naive"
#define CUDA_NAIVE "cudaNaive"
#define CUDA_DP "cudaDP"

// Default output values.
#define OUT_DEFAULT "output/out.png"
#define NAIVE_OUT_DEFAULT "output/naive.png"
#define CUDA_NAIVE_OUT_DEFAULT "output/cudaNaive.png"
#define CUDA_DP_OUT_DEFAULT "output/cudaDP.png"

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

