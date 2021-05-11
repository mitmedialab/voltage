CXX       = nvcc
CXXFLAGS  = -O3 -std=c++11

LDFLAGS   = -Xcompiler -fopenmp
LDLIBS    = -lgomp -lm -lpthread -lcudart -lnppif

default: libpreprocess.a

.SUFFIXES: .c .cpp .cu .o

libpreprocess.a:  motion.o wrapper.o pre_unet.o
	ar rcs $@ $^ 

.c.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


.cpp.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@

.cu.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


clean:
	rm -f *.o *.a
