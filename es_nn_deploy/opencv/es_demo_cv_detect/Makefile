CROSS_COMPILE ?=
OBJS ?= es_demo_cv_detect.o
CC ?= $(CROSS_COMPILE)gcc
CXX ?= $(CROSS_COMPILE)g++
LD ?= $(CROSS_COMPILE)ld
OUT_BIN ?= es_demo_cv_detect
dep_file = .$@.d
LDFLAGS := -lopencv_core -lopencv_imgcodecs -lopencv_videoio \
			-lopencv_imgproc -lopencv_highgui -lopencv_objdetect \
			-lpthread

CFLAGS := -I/usr/local/include/opencv4 -std=c++11
CXXFLAGS :=

all: $(OBJS)
	$(CXX) $(CXXFLAGS)  -o $(OUT_BIN) $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -Wp,-MD,$(dep_file) -c -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf $(OUT_BIN)
	rm -f $(shell find -name "*.d")

