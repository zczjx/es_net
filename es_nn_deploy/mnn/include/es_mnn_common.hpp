#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>

#include <vector>
#include <deque>
#include <iostream>
#include <cstdio>

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"

using namespace std;
namespace es_mnn {
std:: list< std:: pair<MNN::Tensor*, uint8_t> > load_mnist_fmt_data(
    const char *path_dataset, const char *path_labels)
{
    std:: list< std:: pair<MNN::Tensor*, uint8_t> > ret_list;
    FILE *file_dataset;
    FILE *file_labels;
    uint8_t bytes_buf[4], *tmp_img;
    uint32_t dataset_magic = 0, labels_magic = 0;
    uint32_t num_dataset = 0, num_labels = 0;
    int height = 0, width = 0;
    size_t ret, img_size;
    int i, j;
    MNN::CV::ImageProcess::Config img_cfg;
    MNN::CV::ImageProcess *img_process = NULL;

    img_cfg.filterType = MNN::CV::Filter::NEAREST;
    img_cfg.sourceFormat = MNN::CV::ImageFormat::GRAY;
    img_cfg.destFormat = MNN::CV::ImageFormat::GRAY;
    img_process = MNN::CV::ImageProcess::create(img_cfg);

    file_dataset = fopen(path_dataset, "rb");
    file_labels = fopen(path_labels, "rb");

    std:: cout << "loading mnist data......" << std:: endl;
    ret = fread(&dataset_magic, 1, 4, file_dataset);
    dataset_magic = ntohl(dataset_magic);
    printf("dataset_magic: 0x%08x\n", dataset_magic);

    ret = fread(&num_dataset, 1, 4, file_dataset);
    num_dataset = ntohl(num_dataset);
    printf("num_dataset: 0x%08x\n", num_dataset);
   
    ret = fread(&height, 1, 4, file_dataset);
    height = ntohl(height);
    printf("height: 0x%08x\n", height);

    ret = fread(&width, 1, 4, file_dataset);
    width = ntohl(width);
    printf("width: 0x%08x\n", width);

    ret = fread(&labels_magic, 1, 4, file_labels);
    labels_magic = ntohl(labels_magic);
    printf("labels_magic: 0x%08x\n", labels_magic);

    ret = fread(&num_labels, 1, 4, file_labels);
    num_labels = ntohl(num_labels);
    printf("num_labels: 0x%08x\n", num_labels);

    img_size = height * width;
    printf("img_size: %lu\n", img_size);

    tmp_img = new uint8_t(img_size);
    std::vector<int> dims{1, height, width, 1};

    for(i = 0; i < num_labels; i++)
    {
        uint8_t label;
        MNN::Tensor *tmp_tensor = NULL;
        uint8_t *nhwc_data = NULL;
        int nhwc_size;
        halide_type_t btype;
        
        tmp_tensor = MNN::Tensor::create<uint8_t>(dims, NULL, MNN::Tensor::TENSORFLOW);
        // cout << "nhwc_size: " << nhwc_size << endl;
        nhwc_data   = tmp_tensor->host<uint8_t>();
        ret = fread(nhwc_data, 1, img_size, file_dataset);
        // img_process->convert(tmp_img, width, height, 0, tmp_tensor);

    /*********
        btype = tmp_tensor->getType();
        printf("btype.code: %d\n", btype.code);
        printf("btype.bits: %d\n", btype.bits);
        printf("btype.lanes: %d\n", btype.lanes);
    ***/
        // cout << "tmp_tensor->size(): " << tmp_tensor->size() << endl;
        // cout << "tmp_tensor->elementSize(): " << tmp_tensor->elementSize() << endl;
        // img_process->convert(tmp_img, width, height, 0, tmp_tensor);
        ret = fread(&label, 1, sizeof(label), file_labels);
        ret_list.push_back(std:: make_pair(tmp_tensor, label));
    }

    fclose(file_dataset);
    fclose(file_labels);
    std:: cout << "finish loading " << num_labels << " items"  << std:: endl;
    std:: cout << std:: endl;

    return ret_list;
}

}