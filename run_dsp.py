import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import pandas as pd
import time
from ray import tune
import ray
ray.init()
from neuralforecast.auto import AutoPatchTST_with_DSP
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss, MQLoss
from neuralforecast.losses.numpy import mae, mse

# from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo
from datasetsforecast.long_horizon2 import LongHorizon2, LongHorizon2Info

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# command line:
# python run_dsp.py --dataset 'ETTh1' --horizon 96 --num_samples 20

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
      "learning_rate": tune.choice([1e-4]),                 
      "max_steps": tune.choice([300]),                                  
      "input_size": input_size,    
      "loss": None,                                        
      "n_pool_kernel_size": tune.choice([[16, 8, 1]]),               
      "n_freq_downsample": tune.choice([[16, 8, 1]]),                         
      "dropout_prob_theta": tune.choice([0.2]),                                
      "activation": tune.choice(['ReLU']),                                     
      "n_blocks":  tune.choice([[1, 1, 1]]),                                  
      "mlp_units":  tune.choice([[[32, 32]] * 3]),        
      "interpolation_mode": tune.choice(['linear']),      

      "val_check_steps": tune.choice([20]),                                   
      "encoder_layers": tune.choice([1, 3]),
      "n_heads": tune.choice([2,4,8]),
      "hidden_size": tune.choice([32, 64, 128]),
      "linear_hidden_size": tune.choice([256]),
      "fc_dropout": tune.choice([0.2]),
      "head_dropout": tune.choice([0.2]),
      "dropout": tune.choice([0.2]),
      "revin": tune.choice([True]),
      "batch_size": tune.choice([32]),                                                                        
      "scaler_type": tune.choice([None, "robust", "standard"]),  
      "step_size": tune.choice([1]),
      "transformer_input_size": tune.choice([96]),
      "batch_normalization": tune.choice([False]),
      "drop_last_loader": tune.choice([False]),
    } 

    #### 2. 选择搜索算法
    from ray.tune.search.optuna import OptunaSearch

    #### 3. 确定模型
    models = [
        AutoPatchTST_with_DSP(
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
        df=Y_df, val_size=val_size, test_size=test_size,step_size=1, n_windows=None
    )

    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df["AutoPatchTST_with_DSP"].values
    n_series = len(Y_df.unique_id.unique())
    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)
    print("\n" * 4)
    print("Parsed results")
    print(f"AutoPatchTST_with_DSP {dataset} h={horizon}")
    print("test_size", test_size)
    print("y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
    print("y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)
    # print(' best validation hyperparameter:\t', nf.models[0].results.get_best_result().config)
    print("MSE: ", mse(y_hat, y_true))
    print("MAE: ", mae(y_hat, y_true))
    
    # Save Outputs
    # if not os.path.exists(f"./data/{dataset}"):
    #     os.makedirs(f"./data/{dataset}")
    # yhat_file = f"./data/{dataset}/{horizon}_mix_forecasts.csv"
    # Y_hat_df.to_csv(yhat_file, index=False)
