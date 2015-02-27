
CC=g++
LD=g++
CFLAGS=-c -g -Wno-deprecated -O3 -finline-functions 
INCLUDE=-I /home/cristina/eigen/Eigen
NVCC=nvcc
NVCCFLAGS = -c -g -O3

all: ParEGOIteration8

ParEGOIteration8:  Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o ParEGOIteration8.o Matrix.o Cuda_Utilities.o
	$(LD) -L/usr/local/cuda-6.5/lib64 Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o ParEGOIteration8.o Matrix.o Cuda_Utilities.o -lcublas -lcudart -o ParEGOIteration8

Cuda_Utilities.o: Cuda_Utilities.cpp
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) -c Cuda_Utilities.cpp -lcublas -lcudart

Matrix.o: Matrix.cpp 
	$(CC) $(INCLUDE) $(CFLAGS) -c Matrix.cpp

Utilities.o: Utilities.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c Utilities.cpp

WeightVector.o: WeightVector.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c WeightVector.cpp

SearchSpace.o: SearchSpace.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c SearchSpace.cpp

DACE.o: DACE.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c DACE.cpp

GeneticAlgorithm.o: GeneticAlgorithm.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c GeneticAlgorithm.cpp

ParEGOIteration8.o: ParEGOIteration8.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c ParEGOIteration8.cpp

clean:
	rm *o ParEGOIteration8
