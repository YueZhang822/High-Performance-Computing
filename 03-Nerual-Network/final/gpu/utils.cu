#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"

void read_image(char* path, float** image, int image_nums) {
    int size;
    FILE* fp = fopen(path, "rb");
    // skip_header(fp, IMAGE_HEADER_LENGTH);
    int length;
    unsigned int* header = (unsigned int*) malloc(IMAGE_HEADER_LENGTH * sizeof(unsigned int));
    length = fread(header, sizeof(unsigned int), IMAGE_HEADER_LENGTH, fp);
    assert(length == IMAGE_HEADER_LENGTH);
    free(header);

    unsigned char* data = (unsigned char*) malloc(IMAGE_SIZE * sizeof(unsigned char));
    for (int i=0; i<image_nums; i++) {
        size = fread(data, sizeof(unsigned char), IMAGE_SIZE, fp);
        assert(size == IMAGE_SIZE);
        for (int j=0; j<IMAGE_SIZE; j++) {
            assert(data[j] <= MAX_BRIGHTNESS);
            image[i][j] = (float)data[j] / MAX_BRIGHTNESS;
        }
    }
    free(data);
    fclose(fp);
}

void read_label(char* path, int* label, int label_nums) {
    int size;
    FILE* fp = fopen(path, "rb");
    // skip_header(fp, LABEL_HEADER_LENGTH);
    int length;
    unsigned int* header = (unsigned int*) malloc(LABEL_HEADER_LENGTH * sizeof(unsigned int));
    length = fread(header, sizeof(unsigned int), LABEL_HEADER_LENGTH, fp);
    assert(length == LABEL_HEADER_LENGTH);
    free(header);

    unsigned char* data = (unsigned char*) malloc(label_nums * LABEL_SIZE * sizeof(unsigned char));
    size = fread(data, sizeof(unsigned char), label_nums, fp);
    assert(size == label_nums);
    for (int i=0; i<label_nums; i++) {
        assert(data[i] <= 9);
        label[i] = (int)data[i];
    }
    free(data);
    fclose(fp);
}

void read_data_set(float** data_set, char* data_path, int* labels, char* label_path, int data_set_size) {
    for (int i = 0; i < data_set_size; i++) {
        assert(cudaMallocHost((void**) &data_set[i], sizeof(float) * IMAGE_SIZE) == cudaSuccess);
    }
    read_image(data_path, data_set, data_set_size);
    read_label(label_path, labels, data_set_size);
}

void free_data_set(float** data_set, int* labels, int data_set_size) {
    for (int i = 0; i < data_set_size; i++) {
        assert(cudaFreeHost(data_set[i]) == cudaSuccess);
    }
    assert(cudaFreeHost(data_set) == cudaSuccess);
    assert(cudaFreeHost(labels) == cudaSuccess);
}

float gaussian(float mu, float sigma) {
    const float epsilon = 1e-10;

    static float z1;
    static int generate = 0;
    generate = !generate;

    if (!generate)
       return z1 * sigma + mu;

    float u1, u2;
    do {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);

    float z0;
    z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    return z0 * sigma + mu;
}

void kaiming_init(float *data, int m, int n) {
    float std = sqrt(2.0 / n);
    for(int i = 0; i < m * n; i++) {
        data[i] = gaussian(0, std);
    }
}

__device__ float get_val(float* vec, int row, int col, int n) {
    return vec[row * n + col];
}

__device__ void set_val(float* vec, int row, int col, int n, float value) {
    vec[row * n + col] = value;
}

__device__ void add_val(float* vec, int row, int col, int n, float value) {
    vec[row * n + col] += value;
}

int argmax(float* data, int length) {
    int max_idx = 0;
    float max_val = data[0];
    for (int i=1; i<length; i++) {
        if (data[i] > max_val) {
            max_idx = i;
            max_val = data[i];
        }
    }
    return max_idx;
}

float cross_entropy_loss(int y, float* y_hat) {
    return -log(y_hat[y]);
}

__global__ void fc_layer(float* in, float* out, float* w, int w_width, int w_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < w_height && j < w_width) {
        float value = get_val(w, i, j, w_width);
        atomicAdd(&out[i], in[j] * value);
    }
}

__global__ void relu(float* in, int in_dimension, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < in_dimension) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

void softmax(float* in, int in_dimension, float* out, int out_dimension) {
    float sum = 0;
    float max_val = in[0];
    for (int i=1; i<in_dimension; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }
    for (int i=0; i<in_dimension; i++) {
        out[i] = exp(in[i] - max_val);
        sum += out[i];
    }
    for (int i=0; i<in_dimension; i++) {
        out[i] /= sum;
    }
}

__global__ void out_backprop(float* out, float* grad_out, int out_size, int label) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < out_size) {
        grad_out[i] = out[i] - (i==label);
    }
}

__global__ void neuron_backprop(float* in, float* grad_out, float* grad_in, float* w, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        float value = get_val(w, i, j, width);
        atomicAdd(&grad_in[j], value * grad_out[i]);
    }
}

__global__ void relu_backprop(float* in, float* grad_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        grad_in[i] = in[i] > 0 ? grad_in[i] : 0;
    }
}

__global__ void weight_backprop(float* in, float* grad_out, float* grad_w, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        add_val(grad_w, i, j, width, in[j] * grad_out[i]);
    }
}

__global__ void bias_backprop(float* grad_out, float* grad_bias, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        grad_bias[i] += grad_out[i];
    }
}

void allocate_weight_and_data(float** weight_cu, float** bias_cu, float** data_cu, int hidden_size, int num_layers, int data_size) {
    assert(cudaMalloc((void**)weight_cu, IN_DIMENSION * hidden_size * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)bias_cu, hidden_size * sizeof(float)) == cudaSuccess);
    for (int i = 1; i < num_layers-1; ++i) {
        assert(cudaMalloc((void**)(&weight_cu[i]), hidden_size * hidden_size * sizeof(float)) == cudaSuccess);
        assert(cudaMalloc((void**)(&bias_cu[i]), hidden_size * sizeof(float)) == cudaSuccess);
    }
    assert(cudaMalloc((void**)(&weight_cu[num_layers-2]), hidden_size * OUT_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)(&bias_cu[num_layers-2]), OUT_DIMENSION * sizeof(float)) == cudaSuccess);

    for (int i = 0; i < data_size; ++i) {
        assert(cudaMalloc((void**)(&data_cu[i]), IMAGE_SIZE * sizeof(float)) == cudaSuccess);
    }
}

void allocate_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron,
                        float** weight_cu, float** bias_cu, float** data_cu, int hidden_size, int num_layers, int data_size) {
    assert(cudaMalloc((void**)neurons, IN_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)grad_neuron, IN_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)grad_weight, IN_DIMENSION * hidden_size * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)grad_bias, hidden_size * sizeof(float)) == cudaSuccess);

    for (int i = 1; i < num_layers-1; ++i) {
        assert(cudaMalloc((void**)(&neurons[i]), hidden_size * sizeof(float)) == cudaSuccess);
        assert(cudaMalloc((void**)(&grad_neuron[i]), hidden_size * sizeof(float)) == cudaSuccess);
        if (i < num_layers-2) {
            assert(cudaMalloc((void**)(&grad_weight[i]), hidden_size * hidden_size * sizeof(float)) == cudaSuccess);
            assert(cudaMalloc((void**)(&grad_bias[i]), hidden_size * sizeof(float)) == cudaSuccess);
        }
    }

    assert(cudaMalloc((void**)(&neurons[num_layers-1]), OUT_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)(&grad_neuron[num_layers-1]), OUT_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)(&grad_weight[num_layers-2]), hidden_size * OUT_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)(&grad_bias[num_layers-2]), OUT_DIMENSION * sizeof(float)) == cudaSuccess);

    allocate_weight_and_data(weight_cu, bias_cu, data_cu, hidden_size, num_layers, data_size);
}

void free_weight_and_data(float** weight_cu, float** bias_cu, float** data_cu, int num_layers, int data_size) {
    for (int i = 0; i < num_layers-1; ++i) {
        assert(cudaFree(weight_cu[i]) == cudaSuccess);
        assert(cudaFree(bias_cu[i]) == cudaSuccess);
    }
    for (int i = 0; i < data_size; ++i) {
        assert(cudaFree(data_cu[i]) == cudaSuccess);
    }

    assert(cudaFreeHost(weight_cu) == cudaSuccess);
    assert(cudaFreeHost(bias_cu) == cudaSuccess);
    assert(cudaFreeHost(data_cu) == cudaSuccess);
}

void free_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron, float** weight_cu, float** bias_cu, float** data_cu, int num_layers, int data_size) {
    for (int i = 0; i < num_layers; ++i) {
        assert(cudaFree(neurons[i]) == cudaSuccess);
        assert(cudaFree(grad_neuron[i]) == cudaSuccess);
        if (i < num_layers-1) {
            assert(cudaFree(grad_weight[i]) == cudaSuccess);
            assert(cudaFree(grad_bias[i]) == cudaSuccess);
        }
    }

    assert(cudaFreeHost(neurons) == cudaSuccess);
    assert(cudaFreeHost(grad_weight) == cudaSuccess);
    assert(cudaFreeHost(grad_bias) == cudaSuccess);
    assert(cudaFreeHost(grad_neuron) == cudaSuccess);

    free_weight_and_data(weight_cu, bias_cu, data_cu, num_layers, data_size);
}