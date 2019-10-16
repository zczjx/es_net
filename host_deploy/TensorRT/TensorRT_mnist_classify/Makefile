CROSS_COMPILE ?=
OBJS ?= trt_mnist_classify.o logger.o
CC ?= $(CROSS_COMPILE)gcc
CXX ?= $(CROSS_COMPILE)g++
LD ?= $(CROSS_COMPILE)ld
OUT_BIN ?= trt_mnist_classify
dep_file = .$@.d
LDFLAGS := -L/usr/local/cuda/lib64
LDFLAGS += -lpthread -lrt -ldl -lcudnn -lcublas -lcudart -lnvinfer -lnvparsers \
		-lnvinfer_plugin -lnvonnxparser

# STATIC_LDFLAGS = -L../lib -lncnn

CFLAGS := -I../include -std=c++11 \
		-I/usr/local/cuda-10.0/include \
		-L/usr/local/cuda/lib64
CXXFLAGS :=

all: $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(OUT_BIN) $^ $(LDFLAGS) 

%.o: %.cpp
	$(CC) $(CFLAGS) -Wp,-MD,$(dep_file) -c -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf $(OUT_BIN)
	rm -f $(shell find -name "*.d")

