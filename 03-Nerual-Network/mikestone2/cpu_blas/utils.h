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

void read_image(char* path, double** image, int image_nums);

void read_label(char* path, int* label, int label_nums);

void read_data_set(double** data_set, char* data_path, int* labels, char* label_path, int data_set_size);

void free_data_set(double** data_set, int* labels, int data_set_size);

void kaiming_init(double* data, int size, int n);

double get_val(double* vec, int row, int col, int n);

void set_val(double *vec, int row, int col, int n, double value);

void add_val(double* vec, int row, int col, int n, double value);

int argmax(double* data, int len);

double cross_entropy_loss(int y, double* y_hat) ;

void fc_layer(double* in, double* out, double* w, int w_width, int w_height, double* bias);

void relu(double* in, int in_DIMENDION, double* out, int out_size);

void softmax(double* in, int in_DIMENDION, double* out, int out_size);

void allocate_memory(double** neurons, double** grad_weight, double** grad_bias, double** grad_neuron, int hidden_size, int num_layers);

void free_memory(double** neurons, double** grad_weight, double** grad_bias, double** grad_neuron, int num_layers);

#endif