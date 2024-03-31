# Project 3

## Milestone 2

+ learning rate: 0.01
+ epochs: 5
+ GPU threads: 256
+ CPU is tested on MacOS M2
+ GPU is tested on Nvidia A40

### batch size 256

#### training

|                      | CPU w/o BLAS | CPU w/ BLAS | GPU     |
|----------------------|--------------|-------------|---------|
| time(s)              | 463.68       | 161.66      | 79.17   |
| accuracy(%)          | 88.59        | 88.65       | 88.67   |
| grind rate (image/s) | 646.99       | 1855.79     | 3789.44 |
#### test

|                      | CPU w/o BLAS | CPU w/ BLAS | GPU      |
|----------------------|--------------|-------------|----------|
| time(s)              | 6.90         | 1.81        | 0.80     |
| accuracy(%)          | 88.96        | 89.38       | 87.95    |
| grind rate (image/s) | 1450.25      | 5511.28     | 12469.20 |

### batch size 512

#### training

|                      | CPU w/o BLAS | CPU w/ BLAS | GPU     |
|----------------------|--------------|-------------|---------|
| time(s)              | 466.41       | 159.76      | 79.51   |
| accuracy(%)          | 87.24        | 86.99       | 86.95   |
| grind rate (image/s) | 643.21       | 1877.78     | 3772.92 |
#### test

|                      | CPU w/o BLAS | CPU w/ BLAS | GPU      |
|----------------------|--------------|-------------|----------|
| time(s)              | 6.93         | 1.82        | 0.81     |
| accuracy(%)          | 87.80        | 5494.19     | 84.69    |
| grind rate (image/s) | 1443.92      | 87.74       | 12385.08 |

### How to Build?

#### CPU without BLAS

```
make cpu
```

#### CPU with BLAS

```
make blas
```

#### GPU

```
make cuda
```

### How to test?

#### CPU

```
make test-cpu
```

#### GPU

```
make test-cuda
```