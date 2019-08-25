CROSS_COMPILE ?=
OBJS ?= mnn_mnist_classify.o
CC ?= $(CROSS_COMPILE)gcc
CXX ?= $(CROSS_COMPILE)g++
LD ?= $(CROSS_COMPILE)ld
OUT_BIN ?= mnn_mnist_classify
dep_file = .$@.d
LDFLAGS += -lpthread -lopencv_core -lopencv_imgcodecs -lopencv_videoio \
		-lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_dnn \
		-lMNN

STATIC_LDFLAGS = -L../lib

CFLAGS := -I/usr/local/include/opencv4 -I../include -L../lib -std=c++11 \
			-g -fPIC -rdynamic
CXXFLAGS :=

all: $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(OUT_BIN) $^ $(STATIC_LDFLAGS) $(LDFLAGS) 

%.o: %.cpp
	$(CC) $(CFLAGS) -Wp,-MD,$(dep_file) -c -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf $(OUT_BIN)
	rm -f $(shell find -name "*.d")

