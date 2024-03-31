#include "utils.h"

// void skip_header(FILE* fp, int header_length) {
//     int length;
//     unsigned int* header = (unsigned int*) malloc(header_length * sizeof(unsigned int));
//     length = fread(header, sizeof(unsigned int), header_length, fp);
//     assert(length == header_length);
//     free(header);
// }

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
        data_set[i] = (float*) malloc(sizeof(float) * IMAGE_SIZE);
    }
    read_image(data_path, data_set, data_set_size);
    read_label(label_path, labels, data_set_size);
}

void free_data_set(float** data_set, int* labels, int data_set_size) {
    for (int i = 0; i < data_set_size; ++i) {
        free(data_set[i]);
    }
    free(data_set);
    free(labels);
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

float get_val(float* vec, int row, int col, int n) {
    return vec[row * n + col];
}

void set_val(float* vec, int row, int col, int n, float value) {
    vec[row * n + col] = value;
}

void add_val(float* vec, int row, int col, int n, float value) {
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

void fc_layer(float* in, float* out, float* w, int w_width, int w_height, float* bias) {
    for (int i=0; i<w_height; ++i) {
        float sum = 0;
        for (int j=0; j<w_width; ++j) {
            float value = get_val(w, i, j, w_width);
            sum += in[j] * value;
        }
        out[i] = sum + bias[i];
    }
}

void relu(float* in, int in_dimension, float* out, int out_dimension) {
    for (int i=0; i<in_dimension; i++) {
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

void allocate_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron, int hidden_size, int num_layers) {
    neurons[0] = (float*) malloc(IN_DIMENSION * sizeof(float));
    grad_neuron[0] = (float*) malloc(IN_DIMENSION * sizeof(float));
    grad_weight[0] = (float*) malloc(IN_DIMENSION * hidden_size * sizeof(float));
    grad_bias[0] = (float*) malloc(hidden_size * sizeof(float));

    for (int i = 1; i < num_layers-1; ++i) {
        neurons[i] = (float*) malloc(hidden_size * sizeof(float));
        grad_neuron[i] = (float*) malloc(hidden_size * sizeof(float));
        if (i < num_layers-2) {
            grad_weight[i] = (float*) malloc(hidden_size * hidden_size * sizeof(float));
            grad_bias[i] = (float*) malloc(hidden_size * sizeof(float));
        }
    }
    neurons[num_layers-1] = (float*) malloc(OUT_DIMENSION * sizeof(float));
    grad_neuron[num_layers-1] = (float*) malloc(OUT_DIMENSION * sizeof(float));
    grad_weight[num_layers-2] = (float*) malloc(hidden_size * OUT_DIMENSION * sizeof(float));
    grad_bias[num_layers-2] = (float*) malloc(OUT_DIMENSION * sizeof(float));
}

void free_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron, int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        free(neurons[i]);
        free(grad_neuron[i]);
        if (i < num_layers-1) {
            free(grad_weight[i]);
            free(grad_bias[i]);
        }
    }

    free(neurons);
    free(grad_weight);
    free(grad_bias);
    free(grad_neuron);
}