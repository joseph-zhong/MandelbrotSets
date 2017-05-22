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

// // Struct representing complex number.
// struct complexNum {
//   __host__ __device__ complexNum(float a, float bi=0) {
//     this->a = a;
//     this->bi = bi;
//   }
//   float a;
//   float bi;
// };

// /// Inline Operators for complex numbers.
// // Negation operator.
// inline __host__ __device__ complexNum operator-
// (const complexNum &c) { return complexNum(-c.a, -c.bi); }
// 
// // Addition operator.
// inline __host__ __device__ complexNum operator+
// (const complexNum &c0, const complexNum &c1) {
// 	return complexNum(c0.a + c1.a, c0.bi + c1.bi);
// }
// 
// // Subtraction operator.
// inline __host__ __device__ complexNum operator-
// (const complexNum &c0, const complexNum &c1) {
// 	return complexNum(c0.a - c1.a, c0.bi - c1.bi);
// }
// 
// // Absolute Value Squared operator.
// inline __host__ __device__ float absSquared(const complexNum &c0) {
// 	return c0.a * c0.a + c0.bi * c0.bi;
// }
// 
// // Multiplication operator.
// inline __host__ __device__ complexNum operator*
// (const complexNum &c0, const complexNum &c1) {
// 	return complexNum(c0.a * c1.a - c0.bi * c1.bi, c0.bi * c1.a + c0.a * c1.bi);
// }
// 
// // Division operator.
// inline __host__ __device__ complexNum operator/
// (const complexNum &c0, const complexNum &c1) {
// 	float invabs2 = 1 / absSquared(c1);
// 	return complexNum((c0.a * c1.a + c0.bi * c1.bi) * invabs2,
// 								 (c0.bi * c1.a - c1.bi * c0.a) * invabs2);
// }  

// #include "cudaCommon.h"
// 
// // This computes the pixel value using the Escape Time algorithm.
// __device__ char calculatePixelValue(int width, int height, complexNum cMin, complexNum cMax, int x, int y, const float radius);
// 
// // This computes ceil(x / y).
// __host__ __device__ int divup(int x, int y);

// Writes RGB to a ppm FILE*. 
void color(int red, int green, int blue, FILE *fp);

// Parses operation parameters.
void parseArgs(int argc, char *argv[], int *width, int *height, int *maxIterations, char **kernel, char **filename);

#endif //COMMON_H

