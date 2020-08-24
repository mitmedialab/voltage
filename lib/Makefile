MODULES	  := . kalman
INCLUDES  := $(addprefix -I,$(MODULES)) -I /u/ramdas/libs/Tracing/lib_test/

CXX       = g++
CXXFLAGS  = -O3 -std=c++11 $(INCLUDES) 

LDFLAGS   = -fopenmp -L /u/ramdas/libs/Tracing/lib_test/
LDLIBS    = -lgomp -lm -lpthread -llzma -ljbig -ljpeg -lz -ltrace

default: libpreprocess.a

libpreprocess.a:  fft2d/fftsg.o kalman/KalmanFilter.o motion.o wrapper.o
	ar rcs $@ $^ 

.c.o:
	${CXX} -c -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


.cpp.o:
	${CXX} -c -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@

.cu.o:
	${CXX} -c -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS} -o $@


clean:
	rm *.o *.a
