#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>

#include <vector>
#include <deque>
#include <iostream>
#include <list>
#include <cstdio>
#include <algorithm>


namespace es_trt {

typedef struct gray_image {
    float *data;
    size_t bytes;
    int height;
    int width;
} gray_image;

std:: list< std:: pair<gray_image*, uint8_t> > load_mnist_fmt_data(
    const char *path_dataset, const char *path_labels)
{
    std:: list< std:: pair<gray_image*, uint8_t> > ret_list;
    FILE *file_dataset;
    FILE *file_labels;
    uint8_t bytes_buf[4], *tmp_img_buf = NULL;
    uint32_t dataset_magic = 0, labels_magic = 0;
    uint32_t num_dataset = 0, num_labels = 0;
    int height = 0, width = 0;
    size_t ret, img_size;

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

    tmp_img_buf = new uint8_t[img_size];

    for(int i = 0; i < num_labels; i++)
    {
        uint8_t label;
        struct gray_image *tmp_img = NULL;

        tmp_img = new gray_image;
        tmp_img->bytes = img_size * sizeof(float);
        tmp_img->data = new float[tmp_img->bytes];
        tmp_img->height = height;
        tmp_img->width = width;
        ret = fread(tmp_img_buf, 1, img_size, file_dataset);
        for (int j = 0; j < img_size; j++)
        {
            tmp_img->data[j] = float(tmp_img_buf[j]);
        }
            
        ret = fread(&label, 1, sizeof(label), file_labels);
        ret_list.push_back(std:: make_pair(tmp_img, label));
    }

    fclose(file_dataset);
    fclose(file_labels);
    std:: cout << "finish loading " << num_labels << " items"  << std:: endl;
    std:: cout << std:: endl;
    delete [] tmp_img_buf;

    return ret_list;
}

}