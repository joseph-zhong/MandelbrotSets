// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// common.h 
// ---
//
//  Common code.
//

#ifndef COMMON_H
#define COMMON_H

// This checks for a cudaError and exits the program 
// with EXIT_FAILURE if an error was detected.
#define cudaCheck(stmt)                          \
{                                                \
  cudaError_t err = stmt;                        \
  if (err != cudaSuccess) {                      \
    printf("\nERROR ERROR ERROR\n");             \
    printf("ERROR ERROR ERROR\n\n");             \
    printf("\tFailed to run %s\n", #stmt);       \
    gpuAssert((err), __FILE__, __LINE__);        \
    printf("\nERROR ERROR ERROR\n");             \
    printf("ERROR ERROR ERROR\n");               \
    exit(EXIT_FAILURE);                          \
  }                                              \
} 

// Writes RGB to a ppm FILE*. 
void color(int red, int green, int blue, FILE *fp);

// Parses operation parameters.
void parseArgs(int argc, char *argv[], int *width, int *height, int *maxIterations, char **kernel, char **filename);

void saveImage(const char *filename, int *values, int w, int h, int maxIterations);
  
void mapValueToColor(int *r, int *g, int *b, int value, int maxIterations);

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      printf("[gpuAssert]: *** %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

// Struct representing complex number.
struct complexNum {
  __host__ __device__ complexNum(float a, float bi=0) {
    this->a = a;
    this->bi = bi; 
  }
  float a;
  float bi; 
};

/// Inline Operators for complex numbers.
// Negation operator.
inline __host__ __device__ complexNum operator-
(const complexNum &c) { return complexNum(-c.a, -c.bi); }

// Addition operator.
inline __host__ __device__ complexNum operator+
(const complexNum &c0, const complexNum &c1) {
  return complexNum(c0.a + c1.a, c0.bi + c1.bi);
}

// Subtraction operator.
inline __host__ __device__ complexNum operator-
(const complexNum &c0, const complexNum &c1) {
  return complexNum(c0.a - c1.a, c0.bi - c1.bi);
}

// Absolute Value Squared operator.
inline __host__ __device__ float absSquared(const complexNum &c0) {
  return c0.a * c0.a + c0.bi * c0.bi;
}

// Multiplication operator.
inline __host__ __device__ complexNum operator*
(const complexNum &c0, const complexNum &c1) {
  return complexNum(c0.a * c1.a - c0.bi * c1.bi, c0.bi * c1.a + c0.a * c1.bi);
}

// Division operator.
inline __host__ __device__ complexNum operator/
(const complexNum &c0, const complexNum &c1) {
  float invabs2 = 1 / absSquared(c1);
  return complexNum((c0.a * c1.a + c0.bi * c1.bi) * invabs2,
                 (c0.bi * c1.a - c1.bi * c0.a) * invabs2);
}  

 // This computes the pixel value using the Escape Time algorithm.
 // This operation approximately conmputes 35 floating point operations.
__host__  __device__ int calculatePixelValue(int width, int height, int maxIterations, complexNum cMin, complexNum cMax, int x, int y, const float radius);

 // This computes ceil(x / y).
 __host__ __device__ int divup(int x, int y); 

#endif //COMMON_H

