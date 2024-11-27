# MUGST

This is a PyTorch implementation of the paper: Multi-Granularity Spatial-Temporal Graph Convolution Network for Large-Scale Traffic Forecasting

## Repo Structure

```
datasets        ->  raw data and processed data
model           ->  model implementation
utils           ->  dataloader, metrics, meter, ...
```

## Requirements

We implemented MUGST using PyTorch 1.10.0 on Python 3.8.10.  To install all the dependencies, run:

```shell
pip install -r requirements.txt
```

## Data Preparation

You can download the `all_data.zip` file from [Google Drive](https://drive.google.com/file/d/1xWRt71or_8g8CxQlmQkCGylxbWcIFamI/view?usp=drive_link). Unzip the files to the `datasets/` directory:

```shell
unzip /path/to/all_data.zip -d datasets/
```

## Model Training

### PEMS03

```shell
python main.py -d PEMS03 --gpu 0 --log logs/PEMS03_MUGST --sem_gconv --fine --coarse --num_coarse 16 --num_layer 2 --spatial_dim 48
```

### PEMS04

```shell
python main.py -d PEMS04 --gpu 0 --log logs/PEMS04_MUGST --sem_gconv --fine --coarse --num_coarse 16 --num_layer 6 --spatial_dim 64
```

### PEMS07

```shell
python main.py -d PEMS07 --gpu 0 --log logs/PEMS07_MUGST --sem_gconv --fine --coarse --num_coarse 16 --num_layer 8 --spatial_dim 64
```

### PEMS08

```shell
python main.py -d PEMS08 --gpu 0 --log logs/PEMS08_MUGST --sem_gconv --fine --coarse --num_coarse 32 --num_layer 4 --spatial_dim 48
```

### CA2019

```shell
python main.py -d CA2019 --gpu 0 --log logs/CA2019_MUGST --sem_gconv --fine --coarse --num_coarse 32 --num_layer 6 --spatial_dim 80
```

### California

```shell
python main.py -d California --gpu 0 --log logs/California_MUGST --sem_gconv --fine --coarse --num_coarse 32 --num_layer 6 --spatial_dim 80
```

