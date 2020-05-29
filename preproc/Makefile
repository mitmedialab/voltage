
LIBTIFF   := ../lib/tiff-4.1.0_mod/libtiff
FFT2D     := ../lib/fft2d

MODULES   := . util tiff kalman
INCLUDES  := $(addprefix -I,$(MODULES))
INCLUDES  += -I$(LIBTIFF)

CXX       = g++
CXXFLAGS  = -Wall -O2 -std=c++11 -fopenmp $(INCLUDES)

LDFLAGS   = -fopenmp -L$(LIBTIFF)
LDLIBS    = -lm -lpthread $(LIBTIFF)/.libs/libtiff.a -llzma -ljbig -ljpeg -lz

TARGET    = main

SRCS_CC  := $(foreach sdir,$(MODULES),$(wildcard $(sdir)/*.c))
SRCS_CC  += $(FFT2D)/fftsg.c
SRCS_CXX := $(foreach sdir,$(MODULES),$(wildcard $(sdir)/*.cpp))
OBJS      = $(filter %.o,$(SRCS_CC:.c=.o) $(SRCS_CXX:.cpp=.o))


.PHONY: all clean


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) $(TARGET) $(OBJS) core*

