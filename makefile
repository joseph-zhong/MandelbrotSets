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
	nvcc $(FLAGS) $(NVCC_FLAGS) metrics.o common.o cudaCommon.o naive.o cudaNaive.o cudaDP.o main.cu -o main
	
cudaNaive.o: cudaNaive.cu metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) metrics.o -c cudaNaive.cu

cudaDP.o: cudaDP.cu cudaCommon.o common.o metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) cudaCommon.o common.o metrics.o -c cudaDP.cu

naive.o: naive.cu metrics.o common.o
	nvcc $(FLAGS) $(NVCC_FLAGS) metrics.o common.o -c naive.cu

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
	./main 1024 1024 512 naive images/naive.png
	./main 1024 1024 512 cudaNaive images/cudaNaive.png 
	./main 1024 1024 512 cudaDP images/cudaDP.png
	xdg-open images/cudaNaive.png
