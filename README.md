# Long Horizon Forecasting Experiments with Distribution Shift Prediction

<br>

## Introduction

This repository contains the code for the paper "To See the Big Picture: Time Series Forecasting with Distribution Shift Prediction".

The code is based on the [NeuralForecast]
https://github.com/nixtla/neuralforecast

## Key Points

![alt text](https://github.com/Houyikai/distribution-shift-predictor/blob/main/pics/overall%20structure.png)

## Main Results

![alt text](https://github.com/Houyikai/distribution-shift-predictor/blob/main/pics/results.png)

## Reproducibility

1. Create a conda environment `neuralforecast` using the `environment.yml` file.

```shell
conda env create -f environment.yml
```

3. Activate the conda environment using

```shell
conda activate neuralforecast
```

4. Run the experiments for each dataset and each model using with

- `--horizon` parameter in `[96, 192, 336, 720]`
- `--dataset` parameter in `['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']`
  <br>

```shell
python run_dsp.py --dataset 'ETTh1' --horizon 96 --num_samples 10
```

You can access the final forecasts from the `./data/{dataset}/{horizon}_forecasts.csv` file. Example: `./data/ETTh1/96_forecasts.csv`.

Or run the example in notebook test_dsp.

<br><br>
