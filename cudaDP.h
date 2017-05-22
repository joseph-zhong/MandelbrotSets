// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaDP.h
// ---
// 
//  This is the CUDA C implementation with Dynamic Parallelism. 
//

#ifndef CUDA_DP_H
#define CUDA_DP_H

////////////////////////////////
#include <png.h>

#include "cudaCommon.h"

void dwell_color(int *r, int *g, int *b, int dwell);

/** a simple complex type */
struct complex {
	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}
	/** real and imaginary part */
	float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex &a, const complex &b);
inline __host__ __device__ complex operator-
(const complex &a);
inline __host__ __device__ complex operator-
(const complex &a, const complex &b);
inline __host__ __device__ complex operator*
(const complex &a, const complex &b);
inline __host__ __device__ float abs2(const complex &a);
inline __host__ __device__ complex operator/
(const complex &a, const complex &b);

#define MAX_DWELL 512

/** block size along */
#define BSX 64
#define BSY 4

/** maximum recursion depth */
#define MAX_DEPTH 4

/** region below which do per-pixel */
#define MIN_SIZE 32

/** subdivision factor along each axis */
#define SUBDIV 4

/** subdivision when launched from host */
#define INIT_SUBDIV 32

__device__ int pixel_dwell(int w, int h, complexNum cmin, complexNum cmax, int x, int y);

#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__ int same_dwell(int d1, int d2);

__device__ int border_dwell (int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, int d);

__global__ void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell);

__global__ void mandelbrot_pixel_k
(int *dwells, int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, int d);
__device__ void check_error(int x0, int y0, int d);

__global__ void mandelbrot_block_k
(int *dwells, int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, 
 int d, int depth);

void save_image(const char *filename, int *dwells, int w, int h);
    

#define CUT_DWELL (MAX_DWELL / 4)
void dwell_color(int *r, int *g, int *b, int dwell);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

__host__ void cudaDPMandelbrotSets(int height, int width, int maxIterations, 
    const float zoom, const float yPos, const float xPos, const float radius,
    FILE *fp);

// __device__ int commonValue(int v0, int v1, int maxIterations);

// __device__ int calculateBorder(int width, int height, int maxIterations,
//     complexNum cMin, complexNum cMax, int x0, int y0, int size);

// __global__ void pixelKernel(int width, int height, int maxIterations, 
//     complexNum cMin, complexNum cMax, int x0, int y0, int size, int *d_output);

// __global__ void fillKernel(int width, int x0, int y0, int size, int value, int *d_output); 

__global__ void cudaDPMandelbrotSetsKernel(int height, int width, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int depth, const float radius, 
    int *d_output, long long int *d_operations);





#endif // CUDA_DP_H

