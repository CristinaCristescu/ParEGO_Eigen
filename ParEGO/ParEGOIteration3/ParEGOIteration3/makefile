LD = /opt/local/bin/ld

CFLAGS = $(INCLUDE)
TARGET = ParEGO
ParEGO : ParEGOIteration3.o
	$(LD) $(CFLAGS) -o ParEGOIteration3.o
ParEGOIteration3.o : ParEGOIteration3.cpp
	$(CC) $(CFLAGS) -c ParEGOIteration3.cpp




clean: 
	@/bin/rm -f $(TARGET)   
	@/bin/rm *.o

