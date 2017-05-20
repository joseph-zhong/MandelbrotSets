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

#ifndef DEFAULTS_h

#include <stdbool.h>

#define VERBOSE true
#define KERNEL_VERBOSE true

// Default values for specifying size, iterations, and resolution.
#define WIDTH_DEFAULT 600
#define HEIGHT_DEFAULT 400
#define MAX_ITER_DEFAULT 1000
#define X_POS_DEFAULT -0.5
#define Y_POS_DEFAULT 0.0
#define ZOOM_DEFAULT 1.0
#define RADIUS_DEFAULT 4.0

// Default output values.
#define OUT_DEFAULT "images/out.ppm"
#define NAIVE_OUT_DEFAULT "images/naive.ppm"
#define CUDA_NAIVE_OUT_DEFAULT "images/cudaNaive.ppm"
#define CUDA_DP_OUT_DEFAULT "images/cudaDP.ppm"

#define USAGE "Usage:   \n\t%s <width> <height> <maxiter> <output> \n"
#define EXAMPLE "Example: \n\t%s 600 400 1000 images/out.ppm\n\n" 

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

