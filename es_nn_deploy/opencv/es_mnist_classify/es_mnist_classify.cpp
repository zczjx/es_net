#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h> 

#include <vector>
#include <deque>
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "es_cv_common.hpp"

#define US_TO_SEC_UNIT (1000 * 1000)

using namespace cv;
using namespace dnn;
using namespace std;
using namespace es_cv;

static vector<string> classes = {"t-shirt", "trouser", "pullover", "dress", "coat",
                        "sandal", "shirt", "sneaker", "bag", "ankle boot"};

int main(int argc, char** argv)
{
    list< pair<Mat, uint8_t> > test_dat;
    int err_cnt, total_cnt;
    float acc_rate = 0.0;
    Net demo_net;
    time_t start_time, end_time;
    double infer_sec = 0.0;

    if(argc < 4)
    {
        cout << "exam: ./es_demo_cv_detect test_datset test_labels onnx_file" << endl;
        return -1;
    }

    test_dat = load_mnist_fmt_data(argv[1], argv[2]);
    demo_net = readNetFromONNX(argv[3]);
    err_cnt = 0;
    total_cnt = 0;
    start_time = time(NULL);
    for(pair<Mat, uint8_t> pair_item : test_dat)
    {
        Mat blob;
        Mat out_vec;
        int out_val;
        int label = pair_item.second;

        blob = blobFromImage(pair_item.first, 1, Size(28, 28), Scalar(),
                    false, false, CV_8UC1);

        demo_net.setInput(blob);
        out_vec = demo_net.forward();
        minMaxIdx(out_vec.reshape(0, 10), NULL, NULL, NULL, &out_val);
        total_cnt++;
        if(out_val != label)
            err_cnt++;
    }

    end_time = time(NULL);
    infer_sec = difftime(end_time, start_time);
    printf("total_cnt: %d, err_cnt: %d, inference time %f sec\n", 
        total_cnt, err_cnt, infer_sec);
    acc_rate = ((float) (total_cnt - err_cnt)) / (float) (total_cnt);
    printf("test acc_rate: %f\n", acc_rate);
    cout << "finish cnn test......" << endl;
    return 0;
}