
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h> 

#include <vector>
#include <deque>
#include <list>
#include <iostream>
#include <algorithm>
#include <cstdio>

#include "platform.h"
#include "net.h"
#include "es_ncnn_common.hpp"

using namespace std;
using namespace es_ncnn;

int main(int argc, char** argv)
{
    list< pair<ncnn:: Mat, uint8_t> > test_dat;
    int err_cnt, total_cnt;
    float acc_rate = 0.0;
    ncnn::Net demo_net;
    time_t start_time, end_time;
    double infer_sec = 0.0;

    if(argc < 5)
    {
        cout << "exam: ./ncnn_mnist_classify test_datset test_labels ncnn.bin ncnn.param " << endl;
        return -1;
    }

    test_dat = load_mnist_fmt_data(argv[1], argv[2]);
    demo_net.load_param(argv[4]);
    demo_net.load_model(argv[3]);
    
    err_cnt = 0;
    total_cnt = 0;
    start_time = time(NULL);
    for(pair<ncnn:: Mat, uint8_t> pair_item : test_dat)
    {
        ncnn:: Mat ncnn_in = pair_item.first;
        int label = pair_item.second;
        ncnn:: Mat ncnn_out;
        int out_val;

        ncnn::Extractor ex = demo_net.create_extractor();
        ex.input("demo_in_data", ncnn_in);
        ex.extract("demo_out_data", ncnn_out);
        vector<float> out_vec(ncnn_out.w);

        for (int j = 0; j < ncnn_out.w; j++)
        {
            out_vec[j] = ncnn_out[j];
        }
        vector<float>:: iterator max_iter = max_element(out_vec.begin(), 
                                                out_vec.end());
        out_val = distance(out_vec.begin(), max_iter);
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
