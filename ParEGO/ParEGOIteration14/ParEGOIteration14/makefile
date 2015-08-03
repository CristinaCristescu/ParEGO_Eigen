
CC=g++
LD=g++
CFLAGS=-c -g -Wno-deprecated -O3 -finline-functions 
INCLUDE=-I ./Eigen -I /home/cristina/NVIDIA_CUDA-6.5_Samples/common/inc
NVCC=nvcc
NVCCFLAGS = -c -g -O3

all: ParEGOIteration8

ParEGOIteration8:  Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o Cuda_Utilities.o ParEGOIteration8.o Matrix.o 
	$(LD) -L/usr/local/cuda-6.5/lib64 Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o Cuda_Utilities.o ParEGOIteration8.o Matrix.o  -lcublas -lcudart -o ParEGOIteration8

Cuda_Utilities.o: Cuda_Utilities.cpp
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) -c Cuda_Utilities.cpp -lcublas -lcudart

Matrix.o: Matrix.cpp 
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) -c Matrix.cpp -lcublas -lcudart

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
	$(NVCC) $(INCLUDE) $(NVCCFLAGS) -c ParEGOIteration8.cpp -lcublas -lcudart


clean:
	rm *o ParEGOIteration8
