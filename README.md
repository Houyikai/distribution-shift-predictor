# Long Horizon Forecasting Experiments with Distribution Shift Prediction

<br>

## Introduction

This repository contains the code for the paper "Time Series Forecasting with Distribution Shift Prediction".

The code is based on the [NeuralForecast]
https://github.com/nixtla/neuralforecast

## Key Points

This paper proposes a universal Transformer architecture based on stationarization and de-stationarization to address the problem of distribution shift and information utilization bottleneck. It comprises two plug-and-play end-to-end training modules. Moving Normalization Kernel (MNK) is used to stationarize the input sequence of the Transformer, which enhances its predictability and reduces the number of parameters. Distribution Shift Predictor (DSP) explicitly models the dynamics of sequence distribution changes, enabling the approximation to the actual distributions for Transformer predictions.
Implementation of these two modules is in the `./neuralforecast/models/dsp.py` and `./neuralforecast/models/mnk.py` folder. The overall structure's implementation is in the `./neuralforecast/models/patchtst_with_dsp.py` folder.
![alt text](https://github.com/Houyikai/distribution-shift-predictor/blob/main/pics/overall%20structure.png)

## Main Results

forecasting accuracy:
![alt text](https://github.com/Houyikai/distribution-shift-predictor/blob/main/pics/results.png)

computation and memory efficiency:
![alt text](https://github.com/Houyikai/distribution-shift-predictor/blob/main/pics/efficiency.png)

## Reproducibility

1. Create a conda environment `neuralforecast` using the `environment.yml` file.

```shell
conda env create -f environment.yml
```

3. Activate the conda environment using

```shell
conda activate long_horizon
```

then install the `datasetsforecast` package using pip:

```shell
pip install git+https://github.com/Nixtla/datasetsforecast.git
```

4. Run the experiments for each dataset and each model using with

- `--horizon` parameter in `[96, 192, 336, 720]`
- `--dataset` parameter in `['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']`
  <br>

```shell
python run_dsp.py --dataset 'ETTh1' --horizon 96 --num_samples 10
```

You can access the final forecasts from the `./data/{dataset}/{horizon}_forecasts.csv` file. Example: `./data/ETTh1/96_forecasts.csv`.

Or run the example in notebook `test_dsp.ipynb`.

<br><br>
