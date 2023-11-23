import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import pandas as pd

import ray
ray.init()
from ray import tune


from neuralforecast.auto import AutoNHITS, AutoHEncoder
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.pytorch import MAE, HuberLoss, MQLoss
from neuralforecast.losses.numpy import mae, mse

# from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# python run_hencoder.py --dataset 'ETTh1' --horizon 96 --num_samples 20

### 自动调参，开销大
if __name__ == "__main__":
    # Parse execution parameters
    verbose = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-horizon", "--horizon", type=int)
    parser.add_argument("-dataset", "--dataset", type=str)
    parser.add_argument("-num_samples", "--num_samples", default=5, type=int)

    args = parser.parse_args()
    horizon = args.horizon
    dataset = args.dataset
    num_samples = args.num_samples

    # Load dataset
    # Y_df, _, _ = LongHorizon.load(directory='./data/', group=dataset)

    Y_df = LongHorizon2.load(directory="./data/", group=dataset)
    freq = LongHorizon2Info[dataset].freq
    n_time = len(Y_df.ds.unique())

    val_size = LongHorizon2Info[dataset].val_size
    test_size = LongHorizon2Info[dataset].test_size

    # Adapt input_size to available data
    input_size = tune.choice([512])

    #### 1. 定义搜索空间
    mix_config = {
        "h": None,
        
        "decoder_dropout": tune.choice([0, 0.2, 0.6]),

        ### transformer:
        "former_hidden_size": tune.choice([32, 64, 128]),
        "former_linear_size": tune.choice([32, 128, 512]),
        "former_heads": tune.choice([4, 8, 16]),
        "former_encoder_layers": tune.choice([1, 3]),
        "former_dropout": tune.choice([0, 0.2, 0.6]),
        "fc_dropout": tune.choice([0, 0.2, 0.6]),
        
        ### mlp:
        "mlp_block_layers": tune.choice([1, 3]),
        "mlp_blocks": tune.choice([1, 3]),
        "mlp_hidden_size": tune.choice([32, 128, 512]), 
        "n_freq_downsample": tune.choice([[96, 24, 1], [24, 12, 1], [4, 2, 1]]),
        "n_pool_kernel_size": tune.choice([[2, 2, 1], [8, 4, 1], [16, 8, 1]]),
        "mlp_dropout": tune.choice([0, 0.2, 0.6]),
        
        ### basic:
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "loss": None,
        "input_size": input_size,
        "random_seed": tune.randint(1, 10),
        # "val_check_steps": tune.choice([100]),
        "batch_size": tune.choice([32]),
        "windows_batch_size": tune.choice([32]),
        "scaler_type": tune.choice([None, "robust", "standard"]),    
    }

    #### 2. 选择搜索算法
    from ray.tune.search.optuna import OptunaSearch

    #### 3. 确定模型
    models = [
        AutoHEncoder(
            h=horizon,
            loss=MAE(),
            config=mix_config,
            search_alg=OptunaSearch(),
            num_samples=num_samples,
            refit_with_val=True,
        )
    ]

    nf = NeuralForecast(models=models, freq=freq)
    Y_hat_df = nf.cross_validation(
        df=Y_df, val_size=val_size, test_size=test_size, n_windows=None
    )
    
    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df["AutoHEncoder"].values
    n_series = len(Y_df.unique_id.unique())
    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)
    print("\n" * 4)
    print("Parsed results")
    print(f"HEncoder {dataset} h={horizon}")
    print("test_size", test_size)
    print("y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
    print("y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)
    # print(' best validation hyperparameter:\t', nf.models[0].results.get_best_result().config)
    print("MSE: ", mse(y_hat, y_true))
    print("MAE: ", mae(y_hat, y_true))

    # Save Outputs
    if not os.path.exists(f"./data/{dataset}"):
        os.makedirs(f"./data/{dataset}")
    yhat_file = f"./data/{dataset}/{horizon}_forecasts.csv"
Y_hat_df.to_csv(yhat_file, index=False)
