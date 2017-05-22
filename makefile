###
# Joseph Zhong
# josephz@cs.washington.edu
# 19 May 2017
# CSE 599 Final Project
# Instructor: Tanner Schmidt
# Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets
# 
# makefile
# ---
#  This handles compilation for the project. 
#
###

FLAGS=-g -lm -lpng  
NVCC_FLAGS=-arch=sm_35 -rdc=true -lcudadevrt -Xcompiler -fopenmp 


all: main 

main: metrics.o common.o cudaCommon.o naive.o cudaNaive.o cudaDP.o main.cu 
	mkdir -p images
	nvcc $(FLAGS) $(NVCC_FLAGS) metrics.o common.o cudaCommon.o naive.o cudaNaive.o cudaDP.o main2.cu -o main
	# ./main
	# ./main 600 400 1000 cudaNaive images/cudaNaive.ppm 
	./main 600 400 1000 cudaDP images/cudaDP.ppm 
	# xdg-open images/cudaNaive.ppm

cudaNaive.o: cudaNaive.cu metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) metrics.o -c cudaNaive.cu

cudaDP.o: cudaDP.cu cudaCommon.o common.o metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) cudaCommon.o common.o metrics.o -c cudaDP.cu

naive.o: naive.c metrics.o common.o
	g++ $(FLAGS) metrics.o common.o -c naive.c

metrics.o: metrics.c
	g++ $(FLAGS) -c metrics.c 

common.o: common.cu
	nvcc $(FLAGS) $(NVCC_FLAGS) -c common.cu

cudaCommon.o: cudaCommon.cu
	nvcc $(FLAGS) $(NVCC_FLAGS) -c cudaCommon.cu

clean:
	find \( -name '*.out' -or -name '*.o' -or -name '*~' -or -name '*.ppm' \) -delete
	rm main
	rm -r images
	rm *.png

run:
	./main 600 400 1000 cudaDP images/cudaDP.ppm

