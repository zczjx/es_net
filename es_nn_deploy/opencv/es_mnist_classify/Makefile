CROSS_COMPILE ?=
OBJS ?= es_mnist_classify.o
CC ?= $(CROSS_COMPILE)gcc
CXX ?= $(CROSS_COMPILE)g++
LD ?= $(CROSS_COMPILE)ld
OUT_BIN ?= es_mnist_classify
dep_file = .$@.d
LDFLAGS := -lpthread -lopencv_core -lopencv_imgcodecs -lopencv_videoio \
		-lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_dnn

CFLAGS := -I/usr/local/include/opencv4 -I../include -std=c++11
CXXFLAGS :=

all: $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(OUT_BIN) $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -Wp,-MD,$(dep_file) -c -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf $(OUT_BIN)
	rm -f $(shell find -name "*.d")

