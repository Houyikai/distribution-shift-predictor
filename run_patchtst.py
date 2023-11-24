import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import pandas as pd

import ray
ray.init()
from ray import tune


from neuralforecast.auto import AutoPatchTST
from neuralforecast.core import NeuralForecast

from neuralforecast.losses.pytorch import MAE, HuberLoss, MQLoss
from neuralforecast.losses.numpy import mae, mse

# from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# python run_patchtst.py --dataset 'ETTh1' --horizon 96 --num_samples 20

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
    Y_df.drop(columns=["index"], inplace=True)

    # Adapt input_size to available data
    input_size = tune.choice([24,48,96])

    #### 1. 定义搜索空间
    tst_config = {
        "h": None,
        "n_heads": tune.choice([2, 4]),
        "encoder_layers": tune.choice([2, 3]),
        "hidden_size": tune.choice([16, 32]),
        "linear_hidden_size": tune.choice([32, 128, 256]),
        "dropout": tune.choice([0, 0.2]),
        "learning_rate": tune.choice([1e-4]),
        "loss": None,
        "stride": tune.choice([2]),
        "revin": tune.choice([True, False]),
        "input_size": input_size,
        "max_steps": tune.choice([300]),
        "val_check_steps": tune.choice([20]),
        "random_seed": tune.choice([42]),
        "step_size": tune.choice([1]),
        "batch_size": tune.choice([32, 128]),
        "scaler_type": tune.choice([None, "robust", "standard"]),    
    }

    #### 2. 选择搜索算法
    from ray.tune.search.optuna import OptunaSearch

    #### 3. 确定模型
    models = [
        AutoPatchTST(
            h=horizon,
            loss=MAE(),
            config=tst_config,
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
    y_hat = Y_hat_df["AutoPatchTST"].values
    n_series = len(Y_df.unique_id.unique())
    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)
    print("\n" * 4)
    print("Parsed results")
    print(f"AutoPatchTST {dataset} h={horizon}")
    print("test_size", test_size)
    print("y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
    print("y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)
    # print(' best validation hyperparameter:\t', nf.models[0].results.get_best_result().config)
    print("MSE: ", mse(y_hat, y_true))
    print("MAE: ", mae(y_hat, y_true))

    # Save Outputs
    # if not os.path.exists(f"./data/{dataset}"):
    #     os.makedirs(f"./data/{dataset}")
    # yhat_file = f"./data/{dataset}/{horizon}_forecasts.csv"
# Y_hat_df.to_csv(yhat_file, index=False)
