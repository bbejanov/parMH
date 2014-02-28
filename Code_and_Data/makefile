
CC=icc
CXX=icpc
FC=ifort
F77=ifort
F90=ifort
CFLAGS=-O3 -xhost -fp-model precise -openmp -I/cluster/prod/include -fPIC
CXXFLAGS=$(CFLAGS)
FFLAGS=-O3 -xhost -fp-model precise -openmp -I/cluster/prod/include -fPIC
FCFLAGS=$(FFLAGS)
F90FLAGS=$(FFLAGS)


LD=xild
#LDFLAGS=
AR=xiar

LIBS=/cluster/prod/lib/librngstreams.a

libfoobar.so : foobar.o
	$(CC) -shared -static-intel $(LDFLAGS) -o $@ $< $(LIBS)


clean : 
	rm -f *.o *.mod *.so



