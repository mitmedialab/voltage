CXX       = g++
CXXFLAGS  = -O3 -fopenmp -fPIC

LDFLAGS   = -fopenmp
LDLIBS    = -lm -lpthread

TARGET = libpreproc.a

all: $(TARGET)

.SUFFIXES: .c .cpp .cu .o

SRCS_CC  := $(wildcard *.c)
SRCS_CPP := $(wildcard *.cpp)
SRCS_CU  := $(wildcard *.cu)
OBJS  = $(SRCS_CC:.c=.o)
OBJS += $(SRCS_CPP:.cpp=.o)
OBJS += $(SRCS_CU:.cu=.o)

$(TARGET): $(OBJS)
	ar rcs $@ $^

.c.o:
	$(CXX) -c $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

.cu.o:
	$(CXX) -c -Xcompiler $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@


clean:
	rm -f *.o *.a

