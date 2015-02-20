



CC = g++ -Wno-deprecated
#NVCC = nvcc -arch=sm_20
LD = g++ -Wno-deprecated
CFLAGS = $(INCLUDE) -O3 -finline-functions 
#NVCCFLAGS = -x cu --verbose -dlink -I. --compiler-bindir=/usr/bin/g++ -I /home/cristina/eigen/Eigen
TARGET = ParEGOIteration5
ParEGOIteration5 : ParEGOIteration5.o   
	$(LD) $(CFLAGS) -o ParEGOIteration5 ParEGOIteration5.o 

#Cuda_Utilities.o : Cuda_Utilities.cu
#	$(NVCC) $(NVCCFLAGS) -o Cuda_Utilities.cu






ParEGO.o : ParEGOIteration5.cpp Utilities.cpp DACE.cpp SearchSpace.cpp WeightVector.cpp GeneticAlgorithm.cpp
	$(CC) $(CFLAGS) -cpp ParEGOIteration5.cpp Utilities.cpp DACE.cpp SearchSpace.cpp WeightVector.cpp GeneticAlgorithm.cpp

clean: 
	@/bin/rm -f $(TARGET)	
	@/bin/rm *.o
