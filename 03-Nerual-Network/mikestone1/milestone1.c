#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

char TRAIN_IMAGE_FILE[] = "../data/train-images-idx3-ubyte";
char TRAIN_LABEL_FILE[] = "../data/train-labels-idx1-ubyte";
char TEST_IMAGE_FILE[] = "../data/t10k-images-idx3-ubyte";
char TEST_LABEL_FILE[] = "../data/t10k-labels-idx1-ubyte";

int forward(double* image, int image_size, int label, double** weight, double** bias,
            double** neurons, int hidden_size, int num_layers, double* loss) {
    // the first layer
    memcpy(neurons[0], image, image_size * sizeof(double));
    fc_layer(neurons[0], neurons[1], weight[0], image_size, hidden_size, bias[0]);
    sigmoid(neurons[1], hidden_size, neurons[1], hidden_size);

    // hidden layers
    int hidden_layer_num = num_layers - 2;
    for (int i = 1; i < hidden_layer_num; ++i) {
        fc_layer(neurons[i], neurons[i+1], weight[i], hidden_size, hidden_size, bias[i]);
        sigmoid(neurons[i+1], hidden_size, neurons[i+1], hidden_size);
    }

    // the output layer
    int output_layer_idx = num_layers - 1;
    fc_layer(neurons[output_layer_idx-1], neurons[output_layer_idx], weight[output_layer_idx-1],
        hidden_size, OUT_DIMENSION, bias[output_layer_idx-1]);

    sigmoid(neurons[output_layer_idx], OUT_DIMENSION, neurons[output_layer_idx], OUT_DIMENSION);

    int pred = argmax(neurons[output_layer_idx], OUT_DIMENSION);

    *loss = square_loss(label, neurons[output_layer_idx], OUT_DIMENSION);

    return pred;
}

void backward(int y, double** weight, double** bias, double** neuron, double** grad_weight,
            double** grad_bias, double** grad_neuron, int hidden_size, int num_layers) {
    int output_layer_idx = num_layers - 1;
    for (int i = 0; i < OUT_DIMENSION; ++i) {
        double y_hat = neuron[output_layer_idx][i];
        grad_neuron[output_layer_idx][i] = (y_hat - (y == i)) * y_hat * (1.0 - y_hat);
    }

    for (int i = output_layer_idx-1; i >= 0; --i) {
        int width = (i == 0 ? IN_DIMENSION : hidden_size);
        int height = (i == output_layer_idx-1 ? OUT_DIMENSION : hidden_size);
        for (int j = 0; j < width; ++j) {
            double sum = 0;
            for (int k = 0; k < height; ++k) {
                sum += grad_neuron[i+1][k] * get_val(weight[i], k, j, width);
            }
            grad_neuron[i][j] = sum * neuron[i][j] * (1 - neuron[i][j]);
        }

        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                add_val(grad_weight[i], j, k, width, neuron[i][k] * grad_neuron[i+1][j]);
            }
        }

        for (int j = 0; j < height; ++j) {
            grad_bias[i][j] += grad_neuron[i+1][j];
        }
    }
}

void clear_grad(double** grad_weight, double** grad_bias, int hidden_size, int num_layers) {
    memset(grad_weight[0], 0, IN_DIMENSION * hidden_size * sizeof(double));
    memset(grad_bias[0], 0, hidden_size * sizeof(double));

    for (int i = 1; i < num_layers-2; ++i) {
        memset(grad_weight[i], 0, hidden_size * hidden_size * sizeof(double));
        memset(grad_bias[i], 0, hidden_size * sizeof(double));
    }

    memset(grad_weight[num_layers-2], 0, hidden_size * OUT_DIMENSION * sizeof(double));
    memset(grad_bias[num_layers-2], 0, OUT_DIMENSION * sizeof(double));
}

void train(double** train_data, int* train_label, int train_size, double** weight, double** bias,
            int hidden_size, int num_layers, double learning_rate, int epochs,
            int batch_size, double* loss, double* accuracy) {
    double** neurons = (double**) malloc(num_layers * sizeof(double*));
    double** grad_weight = (double**) malloc((num_layers-1) * sizeof(double*));
    double** grad_bias = (double**) malloc((num_layers-1) * sizeof(double*));
    double** grad_neuron = (double**) malloc(num_layers * sizeof(double*));

    allocate_memory(neurons, grad_weight, grad_bias, grad_neuron, hidden_size, num_layers);

    for (int e = 0; e < epochs; ++e) {
        double loss_sum = 0;
        int acc_cnt = 0;
        for (int j = 0; j < train_size; j += batch_size) {
            clear_grad(grad_weight, grad_bias, hidden_size, num_layers);
            for (int i = j; i < j + batch_size; ++i) {
                double loss;
                int idx = i % train_size;
                int y = forward(train_data[idx], IN_DIMENSION, train_label[idx], weight, bias, neurons, hidden_size, num_layers, &loss);
                acc_cnt += (y == train_label[idx]);
                backward(train_label[idx], weight, bias, neurons, grad_weight, grad_bias, grad_neuron, hidden_size, num_layers);
                loss_sum += loss;
            }

            double factor = learning_rate / batch_size;

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
        *accuracy = (double)acc_cnt / train_size;
    }
    free_memory(neurons, grad_weight, grad_bias, grad_neuron, num_layers);
}

void eval(double** test_data, int* test_label, int test_size, double** weight, double** bias,
            int hidden_size, int num_layers, double* loss, double* accuracy) {
    double** neurons = (double**) malloc(num_layers * sizeof(double*));
    neurons[0] = (double*) malloc(IN_DIMENSION * sizeof(double));
    for (int i = 1; i < num_layers-1; ++i) {
        neurons[i] = (double*) malloc(hidden_size * sizeof(double));
    }
    neurons[num_layers-1] = (double*) malloc(OUT_DIMENSION * sizeof(double));

    int cnt = 0;
    double loss_sum = 0;

    for (int i = 0; i < test_size; ++i) {
        int y = forward(test_data[i], IN_DIMENSION, test_label[i], weight, bias, neurons, hidden_size, num_layers, &loss_sum);
        cnt += (y == test_label[i]);
    }

    *loss = loss_sum / test_size;
    *accuracy = (double) cnt / test_size;

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
    double lr = atof(argv[5]);
    int weights_num = nl - 1;

    double** train_set = (double**) malloc(sizeof(double*) * TRAINING_SET_SIZE);
    int* train_labels = (int*) malloc(sizeof(int) * TRAINING_SET_SIZE);
    read_data_set(train_set, TRAIN_IMAGE_FILE, train_labels, TRAIN_LABEL_FILE, TRAINING_SET_SIZE);

    double** test_set = (double**) malloc(sizeof(double*) * TEST_SET_SIZE);
    int* test_labels = (int*) malloc(sizeof(int) * TEST_SET_SIZE);
    read_data_set(test_set, TEST_IMAGE_FILE, test_labels, TEST_LABEL_FILE, TEST_SET_SIZE);

    double** weight = (double**) malloc(sizeof(double*) * weights_num);
    weight[0] = (double*) malloc(sizeof(double) * IMAGE_SIZE * nh);
    random_init(weight[0], IMAGE_SIZE * nh);
    for (int i = 1; i < weights_num-1; ++i) {
        weight[i] = (double*) malloc(sizeof(double) * nh * nh);
        random_init(weight[i], nh * nh);
    }
    weight[weights_num-1] = (double*) malloc(sizeof(double) * nh * OUT_DIMENSION);
    random_init(weight[weights_num-1], nh * OUT_DIMENSION);

    double** bias = (double**) malloc(sizeof(double*) * weights_num);
    for (int i = 0; i < weights_num-1; ++i) {
        bias[i] = (double*) malloc(sizeof(double) * nh);
        random_init(bias[i], nh);
    }
    bias[weights_num-1] = (double*) malloc(sizeof(double) * OUT_DIMENSION);
    random_init(bias[weights_num-1], OUT_DIMENSION);

    double loss = 0;
    double accuracy = 0;

    // Measure running time for training
    clock_t start, end;
    start = clock();
    train(train_set, train_labels, TRAINING_SET_SIZE, weight, bias, nh, nl, lr, ne, nb, &loss, &accuracy);
    end = clock();

    double time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for training: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, ((double)TRAINING_SET_SIZE*ne)/time, loss, accuracy*100);

    loss = 0;
    accuracy = 0;
    // Measure running time for testing
    start = clock();
    eval(test_set, test_labels, TEST_SET_SIZE, weight, bias, nh, nl, &loss, &accuracy);
    end = clock();

    time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for validation: %.2lfs, grind rate: %.2lf, loss: %.2f, accuracy: %.2f%%\n", time, (double)TEST_SET_SIZE/time, loss, accuracy*100);

    free_data_set(train_set, train_labels, TRAINING_SET_SIZE);
    free_data_set(test_set, test_labels, TEST_SET_SIZE);
    for (int i = 0; i < weights_num; ++i) {
        free(weight[i]);
        free(bias[i]);
    }
    free(weight);
    free(bias);
}