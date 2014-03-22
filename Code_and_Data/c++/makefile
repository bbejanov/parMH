
CC=icc
CXX=icpc
FC=ifort
F77=ifort
F90=ifort

my_flags = -g -O0 -xhost -fp-model precise -I/cluster/prod/include -fPIC -mkl=sequential 
#my_flags = -O3 -xhost -fp-model precise -fPIC -I$(MKLROOT)/include \
#           -I/cluster/prod/include 
CFLAGS += -std=gnu99 $(my_flags)
CXXFLAGS += -std=c++11 $(my_flags)
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
	$(CXX) $(CXXFLAGS) -shared $(LDFLAGS) -o $@ $^ $(LIBS)

libstatdist.so : my_stat.o statdistros.o statmh.o mhexamples.o


main : main.cpp libstatdist.so
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS) -L. -Wl,-rpath,. -lstatdist

.PHONY : clean
clean : 
	-rm -f *.o *.mod *.so main



