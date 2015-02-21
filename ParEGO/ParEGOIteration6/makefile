
CC=g++
LD=g++
CFLAGS=-c -Wall -Wno-deprecated -O3 -finline-functions 

all: ParEGOIteration5

ParEGOIteration5: ParEGOIteration5.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o
	$(LD) ParEGOIteration5.o Utilities.o WeightVector.o SearchSpace.o DACE.o GeneticAlgorithm.o -o ParEGOIteration5

Utilities.o: Utilities.cpp
	$(CC) $(CFLAGS) -c Utilities.cpp

WeightVector.o: WeightVector.cpp
	$(CC) $(CFLAGS) -c WeightVector.cpp

SearchSpace.o: SearchSpace.cpp
	$(CC) $(CFLAGS) -c SearchSpace.cpp

DACE.o: DACE.cpp
	$(CC) $(CFLAGS) -c DACE.cpp

GeneticAlgorithm.o: GeneticAlgorithm.cpp
	$(CC) $(CFLAGS) -c GeneticAlgorithm.cpp

ParEGOIteration5.o: ParEGOIteration5.cpp
	$(CC) $(CFLAGS) -c ParEGOIteration5.cpp

clean:
	rm *o ParEGOIteration5