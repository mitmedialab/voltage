MODULES	  := . kalman
INCLUDES  := $(addprefix -I,$(MODULES))

CXX       = nvcc
CXXFLAGS  = -g -std=c++11 $(INCLUDES) 

LDFLAGS   = -Xcompiler -fopenmp 
LDLIBS    = -lgomp -lm -lpthread -llzma -ljbig -ljpeg -lz

default: libpreprocess.a

libpreprocess.a:  fft2d/fftsg.o kalman/KalmanFilter.o motion.o wrapper.o
	ar rcs $@ $^ 

.cpp.o:
	${CXX} -c -Xcompiler -fPIC ${CXXFLAGS} ${LDFLAGS} $< ${LDLIBS}

clean:
	rm *.o *.a
