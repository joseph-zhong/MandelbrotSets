// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaCommon.h
// ---
//
//  Common CUDA C code.
//

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

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
 __device__ int calculatePixelValue(int width, int height, int maxIterations, complexNum cMin, complexNum cMax, int x, int y, const float radius);

 // This computes ceil(x / y).
 __host__ __device__ int divup(int x, int y); 


#endif // CUDA_COMMON_H

