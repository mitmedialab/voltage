CXX       = g++
CXXFLAGS  = -O3 -fopenmp -fPIC

TARGET = libutils.a

all: $(TARGET)

.SUFFIXES: .c .cpp .o

SRCS_CC  := $(wildcard *.c)
SRCS_CPP := $(wildcard *.cpp)
OBJS  = $(SRCS_CC:.c=.o)
OBJS += $(SRCS_CPP:.cpp=.o)

$(TARGET): $(OBJS)
	ar rcs $@ $^

.c.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@


clean:
	rm -f *.o *.a

