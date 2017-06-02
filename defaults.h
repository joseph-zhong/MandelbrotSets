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

#define VERBOSE false
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

// #define X_MIN_DEFAULT -0.75
// #define X_MAX_DEFAULT -0.741
// #define Y_MIN_DEFAULT 0.075
// #define Y_MAX_DEFAULT 0.1


// #define X_MIN_DEFAULT -0.37797980
// #define X_MAX_DEFAULT -0.26686869
// #define Y_MIN_DEFAULT  0.60988165
// #define Y_MAX_DEFAULT  0.73456178

// 0.65243547,0.66478114];
// y in [0.49381968,0.50878414

// #define X_MIN_DEFAULT -0.743643
// #define X_MAX_DEFAULT 0.131825
// #define Y_MIN_DEFAULT 0.00003 
// #define Y_MAX_DEFAULT 5000

#define X_POS_DEFAULT (((X_MAX_DEFAULT) - (X_MIN_DEFAULT)) / 2)
#define Y_POS_DEFAULT (((X_MAX_DEFAULT) - (X_MIN_DEFAULT)) / 2)


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

// Usage and example.
#define USAGE "Usage:   \n\t%s <width> <height> <maxiter> <kernel> <output> \n"
#define EXAMPLE "Example: \n\t%s 600 400 1000 naive output/out.png\n\n" 

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

