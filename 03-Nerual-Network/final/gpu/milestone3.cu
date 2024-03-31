#include <assert.h>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

char TRAIN_IMAGE_FILE[] = "../data/train-images-idx3-ubyte";
char TRAIN_LABEL_FILE[] = "../data/train-labels-idx1-ubyte";
char TEST_IMAGE_FILE[] = "../data/t10k-images-idx3-ubyte";
char TEST_LABEL_FILE[] = "../data/t10k-labels-idx1-ubyte";

int forward(float* image, int image_size, int label, float** weight, float** bias,
            float** neurons, int hidden_size, int num_layers, float* loss, int thread_num) {
    assert(cudaMemcpy(neurons[0], image, image_size * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess);
    for (int i = 0; i < num_layers-1; i++) {
        int width = (i == 0 ? image_size : hidden_size);
        int height = (i == num_layers-2 ? OUT_DIMENSION : hidden_size);

        int block_num = ceil(sqrt((float)thread_num));
        dim3 dim_block(block_num, block_num);
        dim3 dim_grid(ceil((float)height/block_num), ceil((float)width/block_num));
        int thread_num_1d = (height > thread_num ? thread_num : height);
        int block_num_1d = ceil((float)height / thread_num_1d);

        assert(cudaMemcpy(neurons[i+1], bias[i], height * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess);
        fc_layer<<<dim_grid, dim_block>>>(neurons[i], neurons[i+1], weight[i], width, height);

        if (i < num_layers-2) {
            relu<<<block_num_1d, thread_num_1d>>>(neurons[i+1], height, neurons[i+1]);
        }
    }

    float out[OUT_DIMENSION];
    assert(cudaMemcpy(out, neurons[num_layers-1], OUT_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    softmax(out, OUT_DIMENSION, out, OUT_DIMENSION);
    assert(cudaMemcpy(neurons[num_layers-1], out, OUT_DIMENSION * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    *loss = cross_entropy_loss(label, out);

    return argmax(out, OUT_DIMENSION);
}

void backward(int y, float** weight, float** bias, float** neuron, float** grad_weight,
            float** grad_bias, float** grad_neuron, int hidden_size, int num_layers, int thread_num) {
    // output layer
    int output_layer_idx = num_layers - 1;
    int thread_num_1d = (OUT_DIMENSION > thread_num ? thread_num : OUT_DIMENSION);
    int block_num_1d = ceil((float)OUT_DIMENSION / thread_num_1d);
    out_backprop<<<block_num_1d, thread_num_1d>>>(neuron[output_layer_idx], grad_neuron[output_layer_idx], OUT_DIMENSION, y);

    // hidden layers
    for (int i = output_layer_idx-1; i >= 0; i--) {
        int width = (i == 0 ? IN_DIMENSION : hidden_size);
        int height = (i == output_layer_idx-1 ? OUT_DIMENSION : hidden_size);

        int block_num = ceil(sqrt((float)thread_num));
        dim3 dim_block(block_num, block_num);
        dim3 dim_grid(ceil((float)height/block_num), ceil((float)width/block_num));
        thread_num_1d = (height > thread_num ? thread_num : height);
        block_num_1d = ceil((float)height / thread_num_1d);

        neuron_backprop<<<dim_grid, dim_block>>>(neuron[i], grad_neuron[i+1], grad_neuron[i], weight[i], width, height);
        relu_backprop<<<block_num_1d, thread_num_1d>>>(neuron[i], grad_neuron[i], height);
        weight_backprop<<<dim_grid, dim_block>>>(neuron[i], grad_neuron[i+1], grad_weight[i], width, height);
        bias_backprop<<<block_num_1d, thread_num_1d>>>(grad_neuron[i+1], grad_bias[i], height);
    }
}

void clear_grad(float** grad_weight, float** grad_bias, int hidden_size, int num_layers) {
    assert(cudaMemset(grad_weight[0], 0, IN_DIMENSION * hidden_size * sizeof(float)) == cudaSuccess);
    assert(cudaMemset(grad_bias[0], 0, hidden_size * sizeof(float)) == cudaSuccess);

    for (int i = 1; i < num_layers-2; ++i) {
        assert(cudaMemset(grad_weight[i], 0, hidden_size * hidden_size * sizeof(float)) == cudaSuccess);
        assert(cudaMemset(grad_bias[i], 0, hidden_size * sizeof(float)) == cudaSuccess);
    }

    assert(cudaMemset(grad_weight[num_layers-2], 0, hidden_size * OUT_DIMENSION * sizeof(float)) == cudaSuccess);
    assert(cudaMemset(grad_bias[num_layers-2], 0, OUT_DIMENSION * sizeof(float)) == cudaSuccess);
}

__global__ void  update_weights(float* w, float* grad_w, int size, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        w[i] -= factor * grad_w[i];
    }
}

void clear_neurons(float** neurons, int num_layers, int hidden_size) {
    for (int i = 0; i < num_layers; ++i) {
        int size = (i == 0 ? IN_DIMENSION : (i == num_layers-1 ? OUT_DIMENSION : hidden_size));

        assert(cudaMemset(neurons[i], 0, size * sizeof(float)) == cudaSuccess);
    }
}

void train(float** train_data, int* train_label, int train_size, float** weight, float** bias,
            int hidden_size, int num_layers, float learning_rate, int epochs,
            int batch_size, float* loss, float* accuracy, int thread_num) {
    float** weight_cu = NULL;
    float** bias_cu = NULL;
    float** train_data_cu = NULL;

    assert(cudaMallocHost((void**)&weight_cu, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&bias_cu, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&train_data_cu, train_size * sizeof(float*)) == cudaSuccess);

    float** neurons = NULL;
    float** grad_weight = NULL;
    float** grad_bias = NULL;
    float** grad_neuron = NULL;

    assert(cudaMallocHost((void**)&neurons, num_layers * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&grad_weight, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&grad_bias, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&grad_neuron, num_layers * sizeof(float*)) == cudaSuccess);

    allocate_memory(neurons, grad_weight, grad_bias, grad_neuron, weight_cu, bias_cu, train_data_cu, hidden_size, num_layers, train_size);

    for (int i = 0; i < train_size; ++i) {
        assert(cudaMemcpy(train_data_cu[i], train_data[i], IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    assert(cudaMemcpy(weight_cu[0], weight[0], IN_DIMENSION * hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(bias_cu[0], bias[0], hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 1; i < num_layers-2; ++i) {
        assert(cudaMemcpy(weight_cu[i], weight[i], hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(bias_cu[i], bias[i], hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    assert(cudaMemcpy(weight_cu[num_layers-2], weight[num_layers-2], hidden_size * OUT_DIMENSION * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(bias_cu[num_layers-2], bias[num_layers-2], OUT_DIMENSION * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    for (int e = 0; e < epochs; ++e) {
        float loss_sum = 0;
        int acc_cnt = 0;
        for (int j = 0; j < train_size; j += batch_size) {
            clear_grad(grad_weight, grad_bias, hidden_size, num_layers);
            for (int i = j; i < j + batch_size; i++) {
                clear_neurons(neurons, num_layers, hidden_size);
                clear_neurons(grad_neuron, num_layers, hidden_size);
                float loss;
                int idx = i % train_size;
                int y = forward(train_data_cu[idx], IN_DIMENSION, train_label[idx], weight_cu, bias_cu, neurons, hidden_size, num_layers, &loss, thread_num);
                acc_cnt += (y == train_label[idx]);
                backward(train_label[idx], weight_cu, bias_cu, neurons, grad_weight, grad_bias, grad_neuron, hidden_size, num_layers, thread_num);
                loss_sum += loss;

                float factor = learning_rate / batch_size;

                for (int i = 0; i < num_layers-1; ++i) {
                    int width = (i == 0 ? IN_DIMENSION : hidden_size);
                    int height = (i == num_layers-2 ? OUT_DIMENSION : hidden_size);

                    int size = width * height;
                    int thread_num_1d = (size > thread_num ? thread_num : size);
                    int block_num_1d = ceil((float)size / thread_num_1d);
                    update_weights<<<block_num_1d, thread_num_1d>>>(weight_cu[i], grad_weight[i], size, factor);

                    thread_num_1d = (height > thread_num ? thread_num : height);
                    block_num_1d = ceil((float)height / thread_num_1d);
                    update_weights<<<block_num_1d, thread_num_1d>>>(bias_cu[i], grad_bias[i], height, factor);
                }
            }
        }
        *loss = loss_sum / train_size;
        *accuracy = (float)acc_cnt / train_size;
        printf("epochs: %d, loss: %.2f, accuracy: %.2f%%\n", e, *loss, (*accuracy)*100);
    }

    for (int i = 0; i < num_layers-1; ++i) {
        int width = (i == 0 ? IN_DIMENSION : hidden_size);
        int height = (i == num_layers-2 ? OUT_DIMENSION : hidden_size);

        int size = width * height;
        assert(cudaMemcpy(weight[i], weight_cu[i], size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

        size = height;
        assert(cudaMemcpy(bias[i], bias_cu[i], size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    }

    cudaDeviceSynchronize();

    free_memory(neurons, grad_weight, grad_bias, grad_neuron, weight_cu, bias_cu, train_data_cu, num_layers, train_size);
}

void eval(float** test_data, int* test_label, int test_size, float** weight, float** bias,
            int hidden_size, int num_layers, float* loss, float* accuracy, int thread_num) {
    float** neurons = NULL;
    assert(cudaMallocHost(&neurons, num_layers * sizeof(float*)) == cudaSuccess);
    assert(cudaMalloc(&neurons[0], IN_DIMENSION * sizeof(float)) == cudaSuccess);
    for (int i = 1; i < num_layers-1; ++i) {
        assert(cudaMalloc(&neurons[i], hidden_size * sizeof(float)) == cudaSuccess);
    }
    assert(cudaMalloc(&neurons[num_layers-1], OUT_DIMENSION * sizeof(float)) == cudaSuccess);

    float** weight_cu = NULL;
    float** bias_cu = NULL;
    float** test_data_cu = NULL;

    assert(cudaMallocHost((void**)&weight_cu, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&bias_cu, (num_layers-1) * sizeof(float*)) == cudaSuccess);
    assert(cudaMallocHost((void**)&test_data_cu, test_size * sizeof(float*)) == cudaSuccess);

    allocate_weight_and_data(weight_cu, bias_cu, test_data_cu, hidden_size, num_layers, test_size);

    for (int i = 0; i < test_size; ++i) {
        assert(cudaMemcpy(test_data_cu[i], test_data[i], IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    assert(cudaMemcpy(weight_cu[0], weight[0], IN_DIMENSION * hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(bias_cu[0], bias[0], hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 1; i < num_layers-2; ++i) {
        assert(cudaMemcpy(weight_cu[i], weight[i], hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(bias_cu[i], bias[i], hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    assert(cudaMemcpy(weight_cu[num_layers-2], weight[num_layers-2], hidden_size * OUT_DIMENSION * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(bias_cu[num_layers-2], bias[num_layers-2], OUT_DIMENSION * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    int cnt = 0;
    float loss_sum = 0;

    for (int i = 0; i < test_size; ++i) {
        clear_neurons(neurons, num_layers, hidden_size);
        int y = forward(test_data_cu[i], IN_DIMENSION, test_label[i], weight_cu, bias_cu, neurons, hidden_size, num_layers, &loss_sum, thread_num);
        cnt += (y == test_label[i]);
    }

    *loss = loss_sum / test_size;
    *accuracy = (float) cnt / test_size;

    for (int i = 0; i < num_layers; ++i) {
        assert(cudaFree(neurons[i]) == cudaSuccess);
    }
    assert(cudaFreeHost(neurons) == cudaSuccess);
    free_weight_and_data(weight_cu, bias_cu, test_data_cu, num_layers, test_size);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        printf("Usage: %s <layers> <units> <epochs> <batch size> <learning rate> <thread_num>\n", argv[0]);
        return 1;
    }
    int nl = atoi(argv[1]) + 2;
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);
    float lr = atof(argv[5]);
    int weights_num = nl - 1;

    int thread_num = atoi(argv[6]);

    srand(time(NULL));

    float** train_set = NULL;
    int* train_labels = NULL;
    assert(cudaMallocHost((void**) &train_set, sizeof(float*) * TRAINING_SET_SIZE) == cudaSuccess);
    assert(cudaMallocHost((void**) &train_labels, sizeof(int) * TRAINING_SET_SIZE) == cudaSuccess);
    read_data_set(train_set, TRAIN_IMAGE_FILE, train_labels, TRAIN_LABEL_FILE, TRAINING_SET_SIZE);


    float** test_set = NULL;
    int* test_labels = NULL;
    assert(cudaMallocHost((void**) &test_set, sizeof(float*) * TEST_SET_SIZE) == cudaSuccess);
    assert(cudaMallocHost((void**) &test_labels, sizeof(int) * TEST_SET_SIZE) == cudaSuccess);
    read_data_set(test_set, TEST_IMAGE_FILE, test_labels, TEST_LABEL_FILE, TEST_SET_SIZE);

    float** weight = NULL;
    assert(cudaMallocHost((void**) &weight, sizeof(float*) * weights_num) == cudaSuccess);
    assert(cudaMallocHost((void**) &weight[0], sizeof(float) * IMAGE_SIZE * nh) == cudaSuccess);
    kaiming_init(weight[0], nh, IMAGE_SIZE);
    for (int i = 1; i < weights_num-1; i++) {
        assert(cudaMallocHost((void**) &weight[i], sizeof(float) * nh * nh) == cudaSuccess);
        kaiming_init(weight[i], nh, nh);
    }
    assert(cudaMallocHost((void**) &weight[weights_num-1], sizeof(float) * nh * OUT_DIMENSION) == cudaSuccess);
    kaiming_init(weight[weights_num-1], OUT_DIMENSION, nh);

    float** bias = NULL;
    assert(cudaMallocHost((void**) &bias, sizeof(float*) * weights_num) == cudaSuccess);
    for (int i = 0; i < weights_num-1; ++i) {
        assert(cudaMallocHost((void**) &bias[i], sizeof(float) * nh) == cudaSuccess);
        kaiming_init(bias[i], nh, 1);
    }
    assert(cudaMallocHost((void**) &bias[weights_num-1], sizeof(float) * OUT_DIMENSION) == cudaSuccess);
    kaiming_init(bias[weights_num-1], OUT_DIMENSION, 1);

    float loss = 0;
    float accuracy = 0;

    // Measure running time for training
    clock_t start, end;
    start = clock();
    train(train_set, train_labels, TRAINING_SET_SIZE, weight, bias, nh, nl, lr, ne, nb, &loss, &accuracy, thread_num);
    end = clock();

    float time = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for training: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, ((float)TRAINING_SET_SIZE*ne)/time, loss, accuracy*100);

    loss = 0;
    accuracy = 0;
    // Measure running time for testing
    start = clock();
    eval(test_set, test_labels, TEST_SET_SIZE, weight, bias, nh, nl, &loss, &accuracy, thread_num);
    end = clock();

    time = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for validation: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, (float)TEST_SET_SIZE/time, loss, accuracy*100);

    free_data_set(train_set, train_labels, TRAINING_SET_SIZE);
    free_data_set(test_set, test_labels, TEST_SET_SIZE);
    for (int i = 0; i < weights_num; ++i) {
        assert(cudaFreeHost(weight[i]) == cudaSuccess);
        assert(cudaFreeHost(bias[i]) == cudaSuccess);
    }
    assert(cudaFreeHost(weight) == cudaSuccess);
    assert(cudaFreeHost(bias) == cudaSuccess);
}