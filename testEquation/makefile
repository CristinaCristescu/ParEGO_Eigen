MATPACK = /Users/cristina/Documents/Documents/project/matpack
LIBS = $(MATPACK)/matpack.a -lXpm -lX11 -lm -L/usr/X11R6/lib
INCLUDE = -DXPM_INCLUDE="<X11/xpm.h>" -I$(MATPACK)/include

CC = g++ -Wno-deprecated
LD = g++ -Wno-deprecated
CFLAGS = $(INCLUDE) -O3 -finline-functions
TARGET = ssError

ssError : ssError.o
	$(LD) $(CFLAGS) -o ssError ssError.o $(LIBS)
ssError.o : ssError.cpp
	$(CC) $(CFLAGS) -c ssError.cpp

clean: 
