
CC=g++
LD=g++
NVCC=nvcc
CFLAGS=-c -Wno-deprecated -O3 -finline-functions -I/usr/local/cuda-6.5/include 
NVCCFLAGS = -c  

all: ParEGOIteration6

ParEGOIteration6: ParEGOIteration6.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o
	$(LD) -L/usr/local/cuda-6.5/lib64 -lcuda -lcudart ParEGOIteration6.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o Cuda_Utilities.o -lcublas -lcudart -o ParEGOIteration6

Cuda_Utilities.o: Cuda_Utilities.cpp
	$(NVCC) $(NVCCFLAGS) Cuda_Utilities.cpp -lcublas -lcudart

Utilities.o: Utilities.cpp
	$(CC) $(CCFLAGS) -c Utilities.cpp

WeightVector.o: WeightVector.cpp
	$(CC) $(CFLAGS) -c WeightVector.cpp

SearchSpace.o: SearchSpace.cpp
	$(CC) $(CFLAGS) -c SearchSpace.cpp

DACE.o: DACE.cpp
	$(CC) $(CFLAGS) -c DACE.cpp

GeneticAlgorithm.o: GeneticAlgorithm.cpp
	$(CC) $(CFLAGS) -c GeneticAlgorithm.cpp

ParEGOIteration6.o: ParEGOIteration6.cpp
	$(CC) $(CFLAGS) -c ParEGOIteration6.cpp



clean:
	rm *o ParEGOIteration6