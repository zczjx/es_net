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
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logging.h"
#include "logger.h"
#include "buffers.h"
#include "nv_trt_common.h"
#include "es_trt_common.hpp"

using namespace std;
using namespace es_trt;

int main(int argc, char **argv)
{
    list< pair<gray_image*, uint8_t> > test_dat;
    int err_cnt, total_cnt;
    float acc_rate = 0.0;
    time_t start_time, end_time;
    double infer_sec = 0.0;
    bool ret;
    // TensorRT instance
    nvinfer1::IBuilder *trt_builder = NULL;
    nvinfer1::INetworkDefinition *demo_net = NULL;
    nvonnxparser::IParser *parser = NULL;
    nvinfer1::ICudaEngine *cu_engine = NULL;
    nvinfer1::IExecutionContext *infer_ctx = NULL;
    void* cu_io_buffers[2];
    int batchsize = 1;

    if(argc < 4)
    {
        cout << "exam: ./trt_mnist_classify test_datset test_labels model.onnx " << endl;
        return -1;
    }

    test_dat = load_mnist_fmt_data(argv[1], argv[2]);
    trt_builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());

    if(NULL == trt_builder)
    {
        cout << "createInferBuilder failed, line: " << __LINE__ << endl;
        return -1;
    }

    demo_net = trt_builder->createNetwork();

    if(NULL == demo_net)
    {
        cout << "trt_builder->createNetwork() failed, line: " << __LINE__ << endl;
        return -1;
    }

    parser = nvonnxparser::createParser(*demo_net, gLogger.getTRTLogger());

    if(NULL == parser)
    {
        cout << "nvonnxparser::createParser() failed, line: " << __LINE__ << endl;
        return -1;
    }

    ret = parser->parseFromFile(argv[3], static_cast<int>(gLogger.getReportableSeverity()));

    if(false == ret)
    {
        cout << "parser->parseFromFile() failed, line: " << __LINE__ << endl;
        return -1;
    }

    trt_builder->setMaxBatchSize(batchsize);
    trt_builder->setMaxWorkspaceSize(512_MiB);
    trt_builder->allowGPUFallback(true);
    trt_builder->setFp16Mode(false);
    trt_builder->setInt8Mode(false);
    // trt_builder->setStrictTypeConstraints(true);
    cu_engine = trt_builder->buildCudaEngine(*demo_net);

    if(NULL == cu_engine)
    {
        cout << "trt_builder->buildCudaEngine() failed, line: " << __LINE__ << endl;
        return -1;
    }

    infer_ctx = cu_engine->createExecutionContext();

    if(NULL == infer_ctx)
    {
        cout << "cu_engine->createExecutionContext() failed, line: " << __LINE__ << endl;
        return -1;
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    int input_idx, output_idx;

    for (int j = 0; j < cu_engine->getNbBindings(); ++j)
    {
        if (cu_engine->bindingIsInput(j))
            input_idx = j;
        else
            output_idx = j;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&cu_io_buffers[input_idx], batchsize * 28 * 28 * sizeof(float)));
    CHECK(cudaMalloc(&cu_io_buffers[output_idx], batchsize * 10 * sizeof(float)));

    err_cnt = 0;
    total_cnt = 0;
    start_time = time(NULL);

    for(pair<gray_image*, uint8_t> pair_item : test_dat)
    {
        struct gray_image *img_in = pair_item.first;
        int label = pair_item.second;
        float cnn_out[10];
        int out_val;

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        CHECK(cudaMemcpyAsync(cu_io_buffers[input_idx], img_in->data,
                    img_in->bytes, cudaMemcpyHostToDevice, stream));
        infer_ctx->enqueue(batchsize, cu_io_buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(cnn_out, cu_io_buffers[output_idx],
                    10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        out_val = distance(cnn_out, max_element(cnn_out, cnn_out + 10));
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

    for(pair<gray_image*, uint8_t> pair_item : test_dat)
    {
        struct gray_image *img = pair_item.first;
        delete [] img->data;
        delete img;
    }

    cudaStreamDestroy(stream);
    CHECK(cudaFree(cu_io_buffers[input_idx]));
    CHECK(cudaFree(cu_io_buffers[output_idx]));
    parser->destroy();
    cu_engine->destroy();
    demo_net->destroy();
    trt_builder->destroy();

    return 0;
}
