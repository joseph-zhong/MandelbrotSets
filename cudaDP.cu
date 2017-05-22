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
#include "cudaCommon.h"
#include "defaults.h"
#include "metrics.h"

#define MIN_SIZE 32
#define MAX_DEPTH 4
#define DIVIDE_FACTOR 4
#define BLOCK_SIZE 64

__host__ void cudaDPMandelbrotSets(int height, int width, int maxIterations, 
    const float zoom, const float yPos, const float xPos, const float radius, 
    FILE *fp) {
  const int OUTPUT_SIZE = height * width * sizeof(int);
  int *h_output = (int*) malloc(OUTPUT_SIZE);
  long long int *h_operations = (long long int*) calloc(1, sizeof(long long int));

  int *d_output = NULL; 
  long long int *d_operations = NULL;
  cudaCheck(cudaMalloc((void **) &d_output, OUTPUT_SIZE));
  cudaCheck(cudaMalloc((void **) &d_operations, sizeof(long long int)));
  cudaCheck(cudaMemcpy(d_operations, h_operations, sizeof(long long int), cudaMemcpyHostToDevice));

  dim3 gridSize(MIN_SIZE, MIN_SIZE);
  dim3 blockSize(BLOCK_SIZE, DIVIDE_FACTOR);

  clock_t start = clock();

  cudaDPMandelbrotSetsKernel<<<gridSize, blockSize>>>(height, width, maxIterations,
      complexNum(-1.5, -1), complexNum(0.5, 1), 0, 0, width / MIN_SIZE, 1, radius,
      d_output, d_operations);
  cudaCheck(cudaThreadSynchronize());
  
  endClock(start);
  
  cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_operations, d_operations, sizeof(long long int), cudaMemcpyDeviceToHost));

 	// Free device output and operations.
  cudaFree(d_output);
  cudaFree(d_operations);
 
	// Write output and operations.
  // fwrite(h_output, OUTPUT_SIZE, 1, fp);
  save_image("TESTcudaDP.png", h_output, width, height);
  g_operations = *h_operations;
 
  free(h_output);
  free(h_operations);
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
  // int value = NEUT_DWELL;
  for (int pixel = tIdx; pixel < size; pixel += blockSize) {
    for (int boundary = 0; boundary < 4; boundary++) {
      int x = boundary % 2 != 0 ? x0 + pixel : (boundary == 0 ? x0 + size - 1 : x0); 
      int y = boundary % 2 == 0 ? y0 + pixel : (boundary == 1 ? y0 + size - 1 : y0);
      value = commonValue(value, calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius), maxIterations);
    }
  }

  __shared__ int s_output[64 * 4];
  int numThreads = min(size, 64 * 4);
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
    // int outputIndex = CHANNELS * width * y + x * CHANNELS;
    // int pixelValue = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius);
    // d_output[outputIndex] = (char) pixelValue;
    // d_output[outputIndex + 1] = (char) pixelValue;
    // d_output[outputIndex + 2] = (char) 255;
  }
}

__global__ void fillKernel(int width, int x0, int y0, int size, int value, int *d_output) {
  int x = threadIdx.x + blockDim.x * blockDim.y;
  int y = threadIdx.y + blockDim.y * blockDim.y;

  if (x < size && y < size) {
    x += x0;
    y += y0;
    d_output[y * width + x] = value;
    // int outputIndex = CHANNELS * width * y + x * CHANNELS;
    // d_output[outputIndex] = (char) value; 
    // d_output[outputIndex + 1] = (char) value; 
    // d_output[outputIndex + 2] = (char) 255; 
  }
}

__global__ void cudaDPMandelbrotSetsKernel(int height, int width, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int depth, const float radius,
    int *d_output, long long int *d_operations) {

  x0 += size * blockIdx.x;
  y0 += size * blockIdx.y;

  int borderVal = calculateBorder(width, height, maxIterations, cMin, cMax, x0, y0, size, radius); 
  // int borderVal = border_dwell(width, height, cMin, cMax, x0, y0, size);

  if(threadIdx.x == 0 && threadIdx.y == 0) {
    if (borderVal != -1) {
      dim3 fillBlockSize(64, 4);
      dim3 fillGridSize(divup(size, 64), divup(size, 4));
      fillKernel<<<fillGridSize, fillBlockSize>>>(width, x0, y0, size, borderVal, d_output);
      //dwell_fill_k<<<fillGridSize, fillBlockSize>>>(d_output, width, x0, y0, size, borderVal);
    }
    else if (depth + 1 < MAX_DEPTH && size / 4 > MIN_SIZE) {
      dim3 recurseGridSize(4, 4);
      dim3 recurseBlockSize(blockDim.x, blockDim.y);
      cudaDPMandelbrotSetsKernel<<<recurseGridSize, recurseBlockSize>>>(height, width, maxIterations, 
          cMin, cMax, x0, y0, size / 4, depth + 1, radius, d_output, d_operations); 
      // mandelbrot_block_k<<<recurseGridSize, recurseBlockSize>>>
      //  (d_output, width, height, cMin, cMax, x0, y0, size / SUBDIV, depth	+ 1);
    }
    else {
      dim3 pixelGridSize(divup(size, 64), divup(size, 4));
      dim3 pixelBlockSize(64, 4);
      pixelKernel<<<pixelGridSize, pixelBlockSize>>>(width, height, maxIterations,
           cMin, cMax, x0, y0, size, radius, d_output);
      // mandelbrot_pixel_k<<<pixelGridSize, pixelBlockSize>>>
      //   (d_output, width, height, cMin, cMax, x0, y0, size);
    }
  }
}


////////////////////////////////

void dwell_color(int *r, int *g, int *b, int dwell);

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex &a) { return complex(-a.re, -a.im); }
inline __host__ __device__ complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
} 

__device__ int pixel_dwell(int w, int h, complexNum cmin, complexNum cmax, int x, int y) {
	complexNum dc = cmax - cmin;
	float fx = (float)x / w, fy = (float)y / h;
	complexNum c = cmin + complexNum(fx * dc.a, fy * dc.bi);
	int dwell = 0;
	complexNum z = c;
	while(dwell < MAX_DWELL && absSquared(z) < 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}  

__device__ int border_dwell (int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, int d) {
	// check whether all boundary pixels have the same dwell
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int bs = blockDim.x * blockDim.y;
	int comm_dwell = NEUT_DWELL;
	// for all boundary pixels, distributed across threads
	for(int r = tid; r < d; r += bs) {
		// for each boundary: b = 0 is east, then counter-clockwise
		for(int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
			comm_dwell = same_dwell(comm_dwell, dwell);
		}
	}  // for all boundary pixels
	// reduce across threads in the block
	__shared__ int ldwells[BSX * BSY];
	int nt = min(d, BSX * BSY);
	if(tid < nt)
		ldwells[tid] = comm_dwell;
	__syncthreads();
	for(; nt > 1; nt /= 2) {
		if(tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
		__syncthreads();
	}
	return ldwells[0];
}

__global__ void mandelbrot_pixel_k
(int *dwells, int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, int d) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
	}
}


/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = dwell;
	}
}  


__global__ void mandelbrot_block_k
(int *dwells, int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, 
 int d, int depth) {
	x0 += d * blockIdx.x, y0 += d * blockIdx.y;
	int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		if(comm_dwell != DIFF_DWELL) {
			// uniform dwell, just fill
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
		} else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
			// subdivide recursively
			dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
			mandelbrot_block_k<<<grid, bs>>>
				(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth	+ 1);
		} else {
			// leaf, per-pixel kernel
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			mandelbrot_pixel_k<<<grid, bs>>>
				(dwells, w, h, cmin, cmax, x0, y0, d);
		}
		// cucheck_dev(cudaGetLastError());
		//check_error(x0, y0, d);
	}
}  // mandelbrot_block_k




/*******************/

// /** find the dwell for the pixel */
// __device__ int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y) {
// 	complex dc = cmax - cmin;
// 	float fx = (float)x / w, fy = (float)y / h;
// 	complex c = cmin + complex(fx * dc.re, fy * dc.im);
// 	int dwell = 0;
// 	complex z = c;
// 	while(dwell < MAX_DWELL && abs2(z) < 2 * 2) {
// 		z = z * z + c;
// 		dwell++;
// 	}
// 	return dwell;
// }  // pixel_dwell

// /** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
// 		element, -1 = dwells are different */
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__ int same_dwell(int d1, int d2) {
	if(d1 == d2)
		return d1;
	else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
		return min(d1, d2);
	else
		return DIFF_DWELL;
} 

// /** evaluates the common border dwell, if it exists */
// __device__ int border_dwell (int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
// 	// check whether all boundary pixels have the same dwell
// 	int tid = threadIdx.y * blockDim.x + threadIdx.x;
// 	int bs = blockDim.x * blockDim.y;
// 	int comm_dwell = NEUT_DWELL;
// 	// for all boundary pixels, distributed across threads
// 	for(int r = tid; r < d; r += bs) {
// 		// for each boundary: b = 0 is east, then counter-clockwise
// 		for(int b = 0; b < 4; b++) {
// 			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
// 			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
// 			int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
// 			comm_dwell = same_dwell(comm_dwell, dwell);
// 		}
// 	}  // for all boundary pixels
// 	// reduce across threads in the block
// 	__shared__ int ldwells[BSX * BSY];
// 	int nt = min(d, BSX * BSY);
// 	if(tid < nt)
// 		ldwells[tid] = comm_dwell;
// 	__syncthreads();
// 	for(; nt > 1; nt /= 2) {
// 		if(tid < nt / 2)
// 			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
// 		__syncthreads();
// 	}
// 	return ldwells[0];
// }  // border_dwell

// /** the kernel to fill the image region with a specific dwell value */
// __global__ void dwell_fill_k
// (int *dwells, int w, int x0, int y0, int d, int dwell) {
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	if(x < d && y < d) {
// 		x += x0, y += y0;
// 		dwells[y * w + x] = dwell;
// 	}
// }  // dwell_fill_k

// /** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
// 		*/
// __global__ void mandelbrot_pixel_k
// (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
// 	int x = threadIdx.x + blockDim.x * blockIdx.x;
// 	int y = threadIdx.y + blockDim.y * blockIdx.y;
// 	if(x < d && y < d) {
// 		x += x0, y += y0;
// 		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
// 	}
// }  // mandelbrot_pixel_k

/** checking for an error */
__device__ void check_error(int x0, int y0, int d) {
	int err = cudaGetLastError();
	if(err != cudaSuccess) {
		printf("error launching kernel for region (%d..%d, %d..%d)\n", 
					 x0, x0 + d, y0, y0 + d);
		assert(0);
	}
}

/** computes the dwells for Mandelbrot image using dynamic parallelism; one
		block is launched per pixel
		@param dwells the output array
		@param w the width of the output image
		@param h the height of the output image
		@param cmin the complex value associated with the left-bottom corner of the
		image
		@param cmax the complex value associated with the right-top corner of the
		image
		@param x0 the starting x coordinate of the portion to compute
		@param y0 the starting y coordinate of the portion to compute
		@param d the size of the portion to compute (the portion is always a square)
		@param depth kernel invocation depth
		@remarks the algorithm reverts to per-pixel Mandelbrot evaluation once
		either maximum depth or minimum size is reached
//  */
// __global__ void mandelbrot_block_k
// (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, 
//  int d, int depth) {
// 	x0 += d * blockIdx.x, y0 += d * blockIdx.y;
// 	int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
// 	if(threadIdx.x == 0 && threadIdx.y == 0) {
// 		if(comm_dwell != DIFF_DWELL) {
// 			// uniform dwell, just fill
// 			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
// 			dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
// 		} else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
// 			// subdivide recursively
// 			dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
// 			mandelbrot_block_k<<<grid, bs>>>
// 				(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth	+ 1);
// 		} else {
// 			// leaf, per-pixel kernel
// 			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
// 			mandelbrot_pixel_k<<<grid, bs>>>
// 				(dwells, w, h, cmin, cmax, x0, y0, d);
// 		}
// 		// cucheck_dev(cudaGetLastError());
// 		//check_error(x0, y0, d);
// 	}
// }  // mandelbrot_block_k


/** save the dwell into a PNG file 
		@remarks: code to save PNG file taken from here 
		  (error handling is removed):
		http://www.labbookpages.co.uk/software/imgProc/libPNG.html
 */
void save_image(const char *filename, int *dwells, int w, int h) {
	png_bytep row;
	
	FILE *fp = fopen(filename, "wb");
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	// exception handling
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	// write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, w, h,
							 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
							 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	// set title
	png_text title_text;
	title_text.compression = PNG_TEXT_COMPRESSION_NONE;
	title_text.key = "Title";
	title_text.text = "Mandelbrot set, per-pixel";
	png_set_text(png_ptr, info_ptr, &title_text, 1);
	png_write_info(png_ptr, info_ptr);

	// write image data
	row = (png_bytep) malloc(3 * w * sizeof(png_byte));
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int r, g, b;
			dwell_color(&r, &g, &b, dwells[y * w + x]);
			row[3 * x + 0] = (png_byte)r;
			row[3 * x + 1] = (png_byte)g;
			row[3 * x + 2] = (png_byte)b;
		}
		png_write_row(png_ptr, row);
	}
	png_write_end(png_ptr, NULL);

  fclose(fp);
  png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  free(row);
}  // save_image


/** gets the color, given the dwell (on host) */
#define CUT_DWELL (MAX_DWELL / 4)
void dwell_color(int *r, int *g, int *b, int dwell) {
	// black for the Mandelbrot set
	if(dwell >= MAX_DWELL) {
		*r = *g = *b = 0;
	} else {
		// cut at zero
		if(dwell < 0)
			dwell = 0;
		if(dwell <= CUT_DWELL) {
			// from black to blue the first half
			*r = *g = 0;
			*b = 128 + dwell * 127 / (CUT_DWELL);
		} else {
			// from blue to white for the second half
			*b = 255;
			*r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
		}
	}
}  // dwell_color

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


