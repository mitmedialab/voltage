CXX       = nvcc
CXXFLAGS  = -O3 -std=c++11 -I${CONDA_PREFIX}/include

LDFLAGS   = -Xcompiler -fopenmp
LDLIBS    = -lgomp -lm -lpthread -L${CONDA_PREFIX}/lib -lcudart -lnppif -lcublas

default: libfastCoordDescent.a

.SUFFIXES: .c .cpp .cu .o

libfastCoordDescent.a:  coord_descent.o wrapper.o
	ar rcs $@ $^ 

.c.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


.cpp.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@

.cu.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


clean:
	rm -f *.o *.a
