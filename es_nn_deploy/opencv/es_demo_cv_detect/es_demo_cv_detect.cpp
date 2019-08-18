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
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "es_cv_common.hpp"

using namespace cv;
using namespace dnn;
using namespace std;
using namespace es_cv;

static vector<string> classes = {"t-shirt", "trouser", "pullover", "dress", "coat",
                        "sandal", "shirt", "sneaker", "bag", "ankle boot"};

int main(int argc, char** argv)
{
    list< pair<Mat, uint8_t> > test_dat;
    int i, j, idx;

    if(argc < 3)
    {
        cout << "exam: ./es_demo_cv_detect test_datset test_labels" << endl;
        return -1;
    }

    test_dat = load_mnist_fmt_data(argv[1], argv[2]);
    idx = atoi(argv[3]);

    j = 0;
    for(pair<Mat, uint8_t> pair_item : test_dat)
    {
        if(j ==  idx)
        {
            Mat img = pair_item.first;
            uint8_t label = pair_item.second;

            cout << "label: " << classes[label] << endl;
            namedWindow("mnist_window", WINDOW_AUTOSIZE);
            resizeWindow("mnist_window", 320, 240);
            imshow("mnist_window", img);
            waitKey(0);
            break;

        }

        j++;

    }


    return 0;
}