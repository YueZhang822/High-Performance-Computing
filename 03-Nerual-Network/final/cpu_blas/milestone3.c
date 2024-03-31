#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#include "utils.h"

char TRAIN_IMAGE_FILE[] = "../data/train-images-idx3-ubyte";
char TRAIN_LABEL_FILE[] = "../data/train-labels-idx1-ubyte";
char TEST_IMAGE_FILE[] = "../data/t10k-images-idx3-ubyte";
char TEST_LABEL_FILE[] = "../data/t10k-labels-idx1-ubyte";

int forward(float* image, int image_size, int label, float** weight, float** bias,
            float** neurons, int hidden_size, int num_layers, float* loss) {
    // the first layer
    memcpy(neurons[0], image, image_size * sizeof(float));
    fc_layer(neurons[0], neurons[1], weight[0], image_size, hidden_size, bias[0]);
    relu(neurons[1], hidden_size, neurons[1], hidden_size);

    // hidden layers
    int hidden_layer_num = num_layers - 2;
    for (int i = 1; i < hidden_layer_num; ++i) {
        fc_layer(neurons[i], neurons[i+1], weight[i], hidden_size, hidden_size, bias[i]);
        relu(neurons[i+1], hidden_size, neurons[i+1], hidden_size);
    }

    // the output layer
    int output_layer_idx = num_layers - 1;
    fc_layer(neurons[output_layer_idx-1], neurons[output_layer_idx], weight[output_layer_idx-1],
        hidden_size, OUT_DIMENSION, bias[output_layer_idx-1]);

    softmax(neurons[output_layer_idx], OUT_DIMENSION, neurons[output_layer_idx], OUT_DIMENSION);

    *loss = cross_entropy_loss(label, neurons[output_layer_idx]);

    return argmax(neurons[output_layer_idx], OUT_DIMENSION);
}

void backward(int y, float** weight, float** bias, float** neuron, float** grad_weight,
            float** grad_bias, float** grad_neuron, int hidden_size, int num_layers) {
    int output_layer_idx = num_layers - 1;
    for (int i = 0; i < OUT_DIMENSION; ++i) {
        grad_neuron[output_layer_idx][i] = (neuron[output_layer_idx][i] - (i == y));
    }

    for (int i = output_layer_idx-1; i >= 0; --i) {
        int width = (i == 0 ? IN_DIMENSION : hidden_size);
        int height = (i == output_layer_idx-1 ? OUT_DIMENSION : hidden_size);
        for (int j = 0; j < width; j++) {
            if (neuron[i][j] > 0) {
                grad_neuron[i][j] = cblas_sdot(height, grad_neuron[i+1], 1, weight[i]+j, width);
            } else {
                grad_neuron[i][j] = 0;
            }
        }

        cblas_sger(CblasRowMajor, height, width, 1.0, grad_neuron[i+1], 1, neuron[i], 1, grad_weight[i], width);

        cblas_saxpy(height, 1.0, grad_neuron[i+1], 1, grad_bias[i], 1);
    }
}

void clear_grad(float** grad_weight, float** grad_bias, int hidden_size, int num_layers) {
    memset(grad_weight[0], 0, IN_DIMENSION * hidden_size * sizeof(float));
    memset(grad_bias[0], 0, hidden_size * sizeof(float));

    for (int i = 1; i < num_layers-2; ++i) {
        memset(grad_weight[i], 0, hidden_size * hidden_size * sizeof(float));
        memset(grad_bias[i], 0, hidden_size * sizeof(float));
    }

    memset(grad_weight[num_layers-2], 0, hidden_size * OUT_DIMENSION * sizeof(float));
    memset(grad_bias[num_layers-2], 0, OUT_DIMENSION * sizeof(float));
}

void train(float** train_data, int* train_label, int train_size, float** weight, float** bias,
            int hidden_size, int num_layers, float learning_rate, int epochs,
            int batch_size, float* loss, float* accuracy) {
    float** neurons = (float**) malloc(num_layers * sizeof(float*));
    float** grad_weight = (float**) malloc((num_layers-1) * sizeof(float*));
    float** grad_bias = (float**) malloc((num_layers-1) * sizeof(float*));
    float** grad_neuron = (float**) malloc(num_layers * sizeof(float*));

    allocate_memory(neurons, grad_weight, grad_bias, grad_neuron, hidden_size, num_layers);

    for (int e = 0; e < epochs; ++e) {
        float loss_sum = 0;
        int acc_cnt = 0;
        for (int j = 0; j < train_size; j += batch_size) {
            clear_grad(grad_weight, grad_bias, hidden_size, num_layers);
            for (int i = j; i < j + batch_size; ++i) {
                float loss;
                int idx = i % train_size;
                int y = forward(train_data[idx], IN_DIMENSION, train_label[idx], weight, bias, neurons, hidden_size, num_layers, &loss);
                acc_cnt += (y == train_label[idx]);
                backward(train_label[idx], weight, bias, neurons, grad_weight, grad_bias, grad_neuron, hidden_size, num_layers);
                loss_sum += loss;
            }

            float factor = learning_rate / batch_size;

            for (int i = 0; i < num_layers-1; ++i) {
                int width = (i == 0 ? IN_DIMENSION : hidden_size);
                int height = (i == num_layers-2 ? OUT_DIMENSION : hidden_size);
                for (int j = 0; j < width * height; ++j) {
                    weight[i][j] -= factor * grad_weight[i][j];
                }
                for (int j = 0; j < height; ++j) {
                    bias[i][j] -= factor * grad_bias[i][j];
                }
            }
        }
        *loss = loss_sum / train_size;
        *accuracy = (float)acc_cnt / train_size;
        printf("epochs: %d, loss: %.2f, accuracy: %.2f%%\n", e, *loss, (*accuracy)*100);
    }
    free_memory(neurons, grad_weight, grad_bias, grad_neuron, num_layers);
}

void eval(float** test_data, int* test_label, int test_size, float** weight, float** bias,
            int hidden_size, int num_layers, float* loss, float* accuracy) {
    float** neurons = (float**) malloc(num_layers * sizeof(float*));
    neurons[0] = (float*) malloc(IN_DIMENSION * sizeof(float));
    for (int i = 1; i < num_layers-1; ++i) {
        neurons[i] = (float*) malloc(hidden_size * sizeof(float));
    }
    neurons[num_layers-1] = (float*) malloc(OUT_DIMENSION * sizeof(float));

    int cnt = 0;
    float loss_sum = 0;

    for (int i = 0; i < test_size; ++i) {
        int y = forward(test_data[i], IN_DIMENSION, test_label[i], weight, bias, neurons, hidden_size, num_layers, &loss_sum);
        cnt += (y == test_label[i]);
    }

    *loss = loss_sum / test_size;
    *accuracy = (float) cnt / test_size;

    for (int i = 0; i < num_layers; ++i) {
        free(neurons[i]);
    }
    free(neurons);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: %s <layers> <units> <epochs> <batch size> <learning rate>\n", argv[0]);
        return 1;
    }
    int nl = atoi(argv[1]) + 2;
    int nh = atoi(argv[2]);
    int ne = atoi(argv[3]);
    int nb = atoi(argv[4]);
    float lr = atof(argv[5]);
    int weights_num = nl - 1;

    srand(time(NULL));

    float** train_set = (float**) malloc(sizeof(float*) * TRAINING_SET_SIZE);
    int* train_labels = (int*) malloc(sizeof(int) * TRAINING_SET_SIZE);
    read_data_set(train_set, TRAIN_IMAGE_FILE, train_labels, TRAIN_LABEL_FILE, TRAINING_SET_SIZE);

    float** test_set = (float**) malloc(sizeof(float*) * TEST_SET_SIZE);
    int* test_labels = (int*) malloc(sizeof(int) * TEST_SET_SIZE);
    read_data_set(test_set, TEST_IMAGE_FILE, test_labels, TEST_LABEL_FILE, TEST_SET_SIZE);

    float** weight = (float**) malloc(sizeof(float*) * weights_num);
    weight[0] = (float*) malloc(sizeof(float) * IMAGE_SIZE * nh);
    kaiming_init(weight[0], nh, IMAGE_SIZE);
    for (int i = 1; i < weights_num-1; ++i) {
        weight[i] = (float*) malloc(sizeof(float) * nh * nh);
        kaiming_init(weight[i], nh, nh);
    }
    weight[weights_num-1] = (float*) malloc(sizeof(float) * nh * OUT_DIMENSION);
    kaiming_init(weight[weights_num-1], OUT_DIMENSION, nh);

    float** bias = (float**) malloc(sizeof(float*) * weights_num);
    for (int i = 0; i < weights_num-1; ++i) {
        bias[i] = (float*) malloc(sizeof(float) * nh);
        kaiming_init(bias[i], nh, 1);
    }
    bias[weights_num-1] = (float*) malloc(sizeof(float) * OUT_DIMENSION);
    kaiming_init(bias[weights_num-1], OUT_DIMENSION, 1);

    float loss = 0;
    float accuracy = 0;

    // Measure running time for training
    clock_t start, end;
    start = clock();
    train(train_set, train_labels, TRAINING_SET_SIZE, weight, bias, nh, nl, lr, ne, nb, &loss, &accuracy);
    end = clock();

    float time = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for training: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, ((float)TRAINING_SET_SIZE*ne)/time, loss, accuracy*100);

    loss = 0;
    accuracy = 0;
    // Measure running time for testing
    start = clock();
    eval(test_set, test_labels, TEST_SET_SIZE, weight, bias, nh, nl, &loss, &accuracy);
    end = clock();

    time = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for validation: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, (float)TEST_SET_SIZE/time, loss, accuracy*100);

    free_data_set(train_set, train_labels, TRAINING_SET_SIZE);
    free_data_set(test_set, test_labels, TEST_SET_SIZE);
    for (int i = 0; i < weights_num; ++i) {
        free(weight[i]);
        free(bias[i]);
    }
    free(weight);
    free(bias);
}