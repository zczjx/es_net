CROSS_COMPILE ?=
OBJS ?= ncnn_mnist_classify.o
CC ?= $(CROSS_COMPILE)gcc
CXX ?= $(CROSS_COMPILE)g++
LD ?= $(CROSS_COMPILE)ld
OUT_BIN ?= ncnn_mnist_classify
dep_file = .$@.d
LDFLAGS += -lpthread -fopenmp

STATIC_LDFLAGS = -L../../lib -lncnn

CFLAGS := -I../include -std=c++11
CXXFLAGS :=

all: $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(OUT_BIN) $^ $(STATIC_LDFLAGS) $(LDFLAGS) 

%.o: %.cpp
	$(CC) $(CFLAGS) -Wp,-MD,$(dep_file) -c -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf $(OUT_BIN)
	rm -f $(shell find -name "*.d")

