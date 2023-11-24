import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import numpy as np

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


POOLING = ["MaxPool1d", "AvgPool1d"]
class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or (
            "cubic" in interpolation_mode
        )
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        if self.interpolation_mode in ["nearest", "linear"]:
            # knots = knots[:,None,:]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            # forecast = forecast[:,0,:]
        elif "cubic" in self.interpolation_mode:
            if self.out_features > 1:
                raise Exception(
                    "Cubic interpolation not available with multiple outputs."
                )
            batch_size = len(backcast)
            knots = knots[:, None, :, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]  # [B,None,H,H] -> [B,H]
            forecast = forecast[:, None, :]  # [B,H] -> [B,None,H]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast
class NHITSBlock_modified(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        futr_input_size: int,
        hist_input_size: int,
        stat_input_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = (
            pooled_hist_size
            + hist_input_size * pooled_hist_size
            + futr_input_size * pooled_futr_size
            + stat_input_size
        )

        self.dropout_prob = dropout_prob
        self.futr_input_size = futr_input_size
        self.hist_input_size = hist_input_size
        self.stat_input_size = stat_input_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )
        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][-1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
        futr_exog: torch.Tensor,
        hist_exog: torch.Tensor,
        stat_exog: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_input_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, hist_exog.reshape(batch_size, -1)), dim=1
            )

        if self.futr_input_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat(
                (insample_y, futr_exog.reshape(batch_size, -1)), dim=1
            )

        if self.stat_input_size > 0:
            insample_y = torch.cat(
                (insample_y, stat_exog.reshape(batch_size, -1)), dim=1
            )

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast

class DistributionShiftPredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        pred_len: int,
        input_len: int,
        layers: int = 3,
        hidden_size: int = 256,
        dropout: float = 0.2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
    ):
        super().__init__()
        futr_input_size = len(futr_exog_list) if futr_exog_list != None else 0
        hist_input_size = len(hist_exog_list) if hist_exog_list != None else 0
        stat_input_size = len(stat_exog_list) if stat_exog_list != None else 0

        # nhits paramaters, no need to tune
        stack_types: list = ["identity"] * 3
        n_pool_kernel_size: list = [16, 8, 1]
        n_freq_downsample: list = [16, 8, 1]
        n_blocks: list = [1] * 3
        pooling_mode: str = "MaxPool1d"
        interpolation_mode: str = "linear"
        activation="ReLU"

        mlp_stack: list = [[hidden_size, hidden_size]] * layers
  
        blocks = self.create_stack(
            h=pred_len,
            input_size=input_len,
            c_out=c_out,
            stack_types=stack_types,
            futr_input_size=futr_input_size,
            hist_input_size=hist_input_size,
            stat_input_size=stat_input_size,
            n_blocks=n_blocks,
            mlp_units=mlp_stack,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout,
            activation=activation,
        )
        self.blocks = torch.nn.ModuleList(blocks)
      
    def create_stack(
        self,
        h,
        input_size,
        c_out,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        futr_input_size,
        hist_input_size,
        stat_input_size,
    ):
        block_list = []
        for i in range(len(stack_types)):    # [identity] * 3
            for block_id in range(n_blocks[i]):
                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + c_out * max(h // n_freq_downsample[i], 1)
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=c_out,
                    interpolation_mode=interpolation_mode,
                )
                nbeats_block = NHITSBlock_modified(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )
                block_list.append(nbeats_block)
        return block_list
    
    # x:[Batch, nvars, input_size]
    def forward(self, x, insample_mask, futr_exog = None, hist_exog=None, stat_exog=None): 


        z = x
        z = z.view(z.shape[0], -1)  # （bs,seq_len)
        residuals = z.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))
        forecast = z[:, -1:, None]  
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            # print(block_forecast.shape) # (bs,h,1)
            residuals = (residuals + backcast) * insample_mask
            forecast = forecast + block_forecast
        forecast = forecast.permute(0,2,1)

        # [Batch, nvars, input_size]
        return forecast