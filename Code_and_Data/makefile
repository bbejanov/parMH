
CC=icc
CXX=icpc
FC=ifort
F77=ifort
F90=ifort

my_flags = -std=gnu99 -g -O0 -xhost -fp-model precise -I/cluster/prod/include -fPIC -mkl=sequential 
#my_flags = -O3 -xhost -fp-model precise -fPIC -I$(MKLROOT)/include \
#           -I/cluster/prod/include 
CFLAGS += $(my_flags)
CXXFLAGS += $(my_flags)
FFLAGS += $(my_flags)
FCFLAGS += $(my_flags)
F90FLAGS += $(my_flags)

LD=xild
AR=xiar


LDFLAGS += -L/cluster/prod/lib -Wl,-rpath,/cluster/prod/lib
LDFLAGS += -mkl=sequential 
#LDFLAGS += -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
#                             $(MKLROOT)/lib/intel64/libmkl_core.a \
#                             $(MKLROOT)/lib/intel64/libmkl_sequential.a \
#                             -Wl,--end-group -lpthread -lm
LIBS += -lrngstreams 

%.so : 
	$(CC) $(CFLAGS) -shared $(LDFLAGS) -o $@ $^ $(LIBS)

libgaussian.so : gaussian.so
	-ln -fs gaussian.so libgaussian.so

gaussian.so : gaussian.o rngs_utils.o mh.o

libfoobar.so : foobar.o

main : main.c libgaussian.so
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS) -L. -Wl,-rpath,. -lgaussian

.PHONY : clean
clean : 
	-rm -f *.o *.mod *.so main



