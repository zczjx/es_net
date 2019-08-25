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
#include <algorithm>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "MNNForwardType.h"
#include "Tensor.hpp"
#include "es_mnn_common.hpp"

using namespace std;
using namespace es_mnn;
using namespace MNN;

int main(int argc, char** argv)
{
    list< pair<MNN:: Tensor*, uint8_t>  > test_dat;
    int err_cnt, total_cnt;
    float acc_rate = 0.0;
    time_t start_time, end_time;
    double infer_sec = 0.0;
    Interpreter *demo_net = NULL;
    ScheduleConfig sched_cfg;
    BackendConfig backend_cfg;
    Session* demo_net_session;
    std::vector<int> dims{1, 1, 28, 28};

    if(argc < 4)
    {
        cout << "exam: ./mnn_mnist_classify test_datset test_labels net.mnn " << endl;
        return -1;
    }

    test_dat = load_mnist_fmt_data(argv[1], argv[2]);
    demo_net = Interpreter:: createFromFile(argv[3]);
    sched_cfg.numThread = 1;
    sched_cfg.type = MNNForwardType::MNN_FORWARD_CPU;
    backend_cfg.precision = BackendConfig::PrecisionMode::Precision_Low;
    backend_cfg.power = BackendConfig::PowerMode::Power_High;
    backend_cfg.memory = BackendConfig::MemoryMode::Memory_High;
    sched_cfg.backendConfig = &backend_cfg;
    demo_net_session = demo_net->createSession(sched_cfg);

    err_cnt = 0;
    total_cnt = 0;
    start_time = time(NULL);

    for(pair<MNN:: Tensor*, uint8_t> pair_item : test_dat)
    {
        Tensor *in_tensor = NULL;
        Tensor *out_tensor = NULL;
        MNN::Tensor *nhwc_Tensor = pair_item.first;
        int label = pair_item.second;
        int out_val;
        float *pout_data = NULL;
        halide_type_t btype;

        in_tensor = demo_net->getSessionInput(demo_net_session, "data");
        in_tensor->setType(4);
        demo_net->resizeTensor(in_tensor, dims);
        // demo_net->resizeTensor(out_tensor, dims);
        // out_tensor->setType(1);
        demo_net->resizeSession(demo_net_session);
        // btype = in_tensor->getType();

        /************
        printf("btype.code: %d\n", btype.code);
        printf("btype.bits: %d\n", btype.bits);
        printf("btype.lanes: %d\n", btype.lanes);
        cout << "in_tensor->size(): " << in_tensor->size() << endl;
        cout << "in_tensor->elementSize(): " << in_tensor->elementSize() << endl;

      
        cout << "in_tensor->size(): " << in_tensor->size() << endl;
        cout << "in_tensor->getDimensionType(): " <<  (int) in_tensor->getDimensionType() << endl; 
        cout << "nhwc_Tensor->getDimensionType(): " <<  (int) nhwc_Tensor->getDimensionType() << endl;
        cout << "out_tensor->getDimensionType(): " <<  (int) out_tensor->getDimensionType() << endl;
        cout << "out_tensor->size(): " << out_tensor->size() << endl;

        ****/

        /***
        std::vector<int> shape = in_tensor->shape();
        cout << "in_tensor->size(): " << in_tensor->size() << endl;
        
        for(int dim: shape)
        {
            cout << dim << " ";
        }
        cout << endl;
        **/

        in_tensor->copyFromHostTensor(nhwc_Tensor);
        // cout << "in_tensor->size(): " << in_tensor->size() << endl;
        // cout << "in_tensor->elementSize(): " << in_tensor->elementSize() << endl;
        demo_net->runSession(demo_net_session);
        out_tensor = demo_net->getSessionOutput(demo_net_session, "dense2_fwd");
        vector<float> out_vec(out_tensor->elementSize());
        pout_data = out_tensor->host<float>();
        
        for (int j = 0; j < out_tensor->elementSize(); j++)
        {
            out_vec[j] = pout_data[j];
        }
        vector<float>:: iterator max_iter = max_element(out_vec.begin(), 
                                                out_vec.end());
        out_val = distance(out_vec.begin(), max_iter);
        total_cnt++;

        // if(total_cnt == 10)
        //    break;

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
