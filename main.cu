#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat load_image(const char *source_image) {
    Mat image = imread(source_image, CV_LOAD_IMAGE_GRAYSCALE);

    return image;
}

int save_image(Mat &image, const char *path) {
    vector<int> output_image_params;

    output_image_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    output_image_params.push_back(100);

    bool is_success = imwrite(path, image, output_image_params);

    if (!is_success) {
        cout << "Could not save image" << endl;
        return -1;
    }

    return 0;
}

void check_arguments(int argc) {
    if(argc < 3) {
        cout << "Run program with following format: ";
        cout << "./binarization_gpu <block_size> <image_name>" << endl;

        return -1;
    }
}

__global__ void sauvolaBinarization(unsigned char *source_image, unsigned char *output_image, int width, int height, int step) {

    int x, y;
    int r = 13;
    float k = 0.12;
    float m, sum, variance, varianceSum, treshold;
    int MAX_STANDARD_DEVIATION = 128;
    int n = (2 * r + 1) * (2 * r + 1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int indexColumn = index % step;
    int delimiter = step * height - r * step;
    int indexRow = index / step;

    if (indexColumn > r && indexColumn < step - r && index > r * step && index < delimiter) {
        sum = 0;
        varianceSum = 0;

        for (y = indexRow - r; y <= indexRow + r; y++) {
            for (x = indexColumn - r; x <= indexColumn + r; x++) {
                sum += source_image[y * step + x];
            }
        }

        m = sum / n;

        for (y = indexRow - r; y <= indexRow + r; y++) {
            for (x = indexColumn - r; x <= indexColumn + r; x++) {
                varianceSum += (source_image[y * step + x] - m) * (source_image[y * step + x] - m);
            }
        }

        variance = sqrt(varianceSum / n);
        treshold = (m * (1.0 + k * (variance / MAX_STANDARD_DEVIATION - 1.0)));

        output_image[index] = (source_image[index] > treshold) ? 255 : 0;
    } else {
        output_image[index] = (source_image[index] > 128) ? 255 : 0;
    }
}

int main(int argc, char **argv) {

    check_arguments(argc);

    int w, h, step;
    float time_difference;

    string infname = "./original_images/" + string(argv[2]) + ".pgm";
    string output_path = "./binarized_images/" + string(argv[2]) + ".pgm";

    Mat source_image = load_image(infname.c_str());
    Mat output_image = source_image.clone();

    if (!source_image.data) {
        cout << "Could not open image" << endl;
        return -1;
    }

    w = source_image.cols;
    h = source_image.rows;
    step = source_image.step;

    int block_size = atoi(argv[1]);
    int image_size = w * h;
    int grid_size = ceil(image_size / block_size);

    unsigned char *gpu_source = nullptr;
    unsigned char *gpu_output = nullptr;

    cudaEvent_t timer_start, timer_stop;
    cudaEventCreate(&timer_start);
    cudaEventCreate(&timer_stop);
    cudaEventRecord(timer_start, 0);

    cudaMalloc<unsigned char>(&gpu_source, h * step);
    cudaMalloc<unsigned char>(&gpu_output, h * step);

    cudaMemcpy(gpu_source, source_image.data, h * step, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_output, output_image.data, h * step, cudaMemcpyHostToDevice);

    sauvolaBinarization << < grid_size, block_size >> > (gpu_source, gpu_output, w, h, step);

    cudaDeviceSynchronize();

    cudaMemcpy(output_image.data, gpu_output, h * step, cudaMemcpyDeviceToHost);

    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop);
    cudaEventElapsedTime(&time_difference, timer_start, timer_stop);

    save_image(output_image, output_path.c_str());

    cout << "Binarization time: " << time_difference << " ms" << endl;

    cudaFree(gpu_source);
    cudaFree(gpu_output);

    return 0;
}