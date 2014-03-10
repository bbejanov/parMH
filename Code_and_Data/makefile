
CC=icc
CXX=icpc
FC=ifort
F77=ifort
F90=ifort
CFLAGS=-O3 -xhost -fp-model precise -openmp -I/cluster/prod/include -fPIC -mkl=parallel
CXXFLAGS=$(CFLAGS)
FFLAGS=-O3 -xhost -fp-model precise -openmp -I/cluster/prod/include -fPIC -mkl=parallel
FCFLAGS=$(FFLAGS)
F90FLAGS=$(FFLAGS)


LD=xild
#LDFLAGS=
AR=xiar

LIBS=/cluster/prod/lib/librngstreams.a

gaussian.so : gaussian.o
	$(CC) $(CFLAGS) -shared -static-intel $(LDFLAGS) -o $@ $< $(LIBS)

libfoobar.so : foobar.o
	$(CC) $(CFLAGS) -shared -static-intel $(LDFLAGS) -o $@ $< $(LIBS)


clean : 
	rm -f *.o *.mod *.so



