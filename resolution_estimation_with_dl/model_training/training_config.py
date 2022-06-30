from typing import Tuple
from dataclasses import dataclass
import torch
from torch.cuda import is_available
from torch.optim import AdamW, Optimizer
from resolution_estimation_with_dl.model_training.training_utils import (
    WeightedMSELoss,
    WeightedMAE,
)


@dataclass
class TrainParameters:
    # Number of iterations for model training
    n_epochs: int = 30
    # The shape of cubes which will be given to a model
    model_input_shape: Tuple[int] = (16, 16, 16)
    # Batch size
    batch_size: int = 400
    # Number of channels in convolution layers
    layer_filters: Tuple[int] = (64, 128, 256, 512, 1024)
    # Parameter `p` for dropout layers
    dropout_rates: Tuple[float] = (0.25, 0.25, 0.25, 0.25, 0.25)
    # Device
    device: str = "cuda:0" if is_available() else "cpu"
    # Upsampling mode and align_corners
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    upsample_mode: str = "trilinear"
    align_corners: bool = True
    # Learning rate for optimizer
    learning_rate: float = 3e-4
    # Which type of regularization to use: `dropout` or `batchnorm`
    regularization: str = "dropout"
    # Whether to initialize the weights with He initialization
    initialize_weights: bool = True
    # Which optimizer to use
    optimizer: Optimizer = AdamW
    # Loss
    criterion: torch.nn.Module = WeightedMSELoss
    # Metric
    metric: torch.nn.Module = WeightedMAE
    # Whether to validate model after each epoch
    validate: bool = True
    # The detailing of the train process
    verbose: int = 2
