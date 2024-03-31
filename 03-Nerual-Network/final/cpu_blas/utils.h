#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define TRAINING_SET_SIZE         60000
#define TEST_SET_SIZE             10000

#define IMAGE_HEADER_LENGTH       4
#define LABEL_HEADER_LENGTH       2

#define IMAGE_SIZE                784
#define LABEL_SIZE                1

#define IN_DIMENSION              784
#define OUT_DIMENSION             10

#define MAX_BRIGHTNESS            255

void skip_header(FILE* fp, int header_length);   // skip the header on top of each image/label

void read_image(char* path, float** image, int image_nums);

void read_label(char* path, int* label, int label_nums);

void read_data_set(float** data_set, char* data_path, int* labels, char* label_path, int data_set_size);

void free_data_set(float** data_set, int* labels, int data_set_size);

void kaiming_init(float* data, int size, int n);

float get_val(float* vec, int row, int col, int n);

void set_val(float *vec, int row, int col, int n, float value);

void add_val(float* vec, int row, int col, int n, float value);

int argmax(float* data, int len);

float cross_entropy_loss(int y, float* y_hat) ;

void fc_layer(float* in, float* out, float* w, int w_width, int w_height, float* bias);

void relu(float* in, int in_DIMENDION, float* out, int out_size);

void softmax(float* in, int in_DIMENDION, float* out, int out_size);

void allocate_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron, int hidden_size, int num_layers);

void free_memory(float** neurons, float** grad_weight, float** grad_bias, float** grad_neuron, int num_layers);

#endif