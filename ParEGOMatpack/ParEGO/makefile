MATPACK = /Users/cristina/projectDev/matpack
LIBS = $(MATPACK)/matpack.a -lXpm -lX11 -lm -L/usr/X11R6/lib
INCLUDE = -DXPM_INCLUDE="<X11/xpm.h>" -I$(MATPACK)/include

CC = /opt/local/bin/g++-mp-4.6 -Wno-deprecated
LD = /opt/local/bin/g++-mp-4.6 -Wno-deprecated
CFLAGS = $(INCLUDE) -O3 -finline-functions 
TARGET = ParEGO
ParEGO : ParEGO.o nrutil.o
	$(LD) $(CFLAGS) -o ParEGO ParEGO.o nrutil.o $(LIBS)	

ParEGO.o : ParEGO.cc
	$(CC) $(CFLAGS) -c ParEGO.cc

nrutil.o : nrutil.c
	$(CC) $(CFLAGS) -c nrutil.c

clean: 
	@/bin/rm -f $(TARGET)	
	@/bin/rm *.o
