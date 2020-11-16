CXX       = nvcc
CXXFLAGS  = -O3 -std=c++14

LDFLAGS   = -Xcompiler -fopenmp
LDLIBS    = -lgomp -lm -lpthread -lcudart -lnppif -lnppc -lcublas

default: libpostprocess.a

.SUFFIXES: .c .cpp .cu .o

libpostprocess.a:  wrapper.o postprocess.o
	ar rcs $@ $^ 

.c.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


.cpp.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@

.cu.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


clean:
	rm -f *.o *.a
