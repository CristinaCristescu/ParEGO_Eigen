
CC=g++
LD=g++
CFLAGS=-c -Wall -Wno-deprecated -O3 -finline-functions 
INCLUDE=-I /Users/cristina/eigen/eigen-eigen-1306d75b4a21/Eigen

all: ParEGOIteration9

ParEGOIteration9: ParEGOIteration9.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o Matrix.o
	$(LD) $(INCLUDE) ParEGOIteration9.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o Matrix.o -o ParEGOIteration9

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

ParEGOIteration9.o: ParEGOIteration9.cpp
	$(CC) $(INCLUDE) $(CFLAGS) -c ParEGOIteration9.cpp

clean:
	rm *o ParEGOIteration9
