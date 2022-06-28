from typing import Tuple
from dataclasses import dataclass
import torch
from torch.cuda import is_available
from torch.optim import AdamW, Optimizer
from ..model_training.training_utils import WeightedMSELoss, WeightedMAE


@dataclass
class TrainParameters:
    n_epoches: int = 30
    model_input_shape: Tuple[int] = (16, 16, 16)
    batch_size: int = 400
    layer_filters: Tuple[int] = (64, 128, 256, 512, 1024)
    dropout_rates: Tuple[float] = (0.25, 0.25, 0.25, 0.25, 0.25)
    device: str = "cuda:0" if is_available() else "cpu"
    upsample_mode: str = "trilinear"
    align_corners: bool = True
    learning_rate: float = 3e-4
    regularization: str = "dropout"
    initialize_weights: bool = True
    optimizer: Optimizer = AdamW
    criterion: torch.nn.Module = WeightedMSELoss
    metric: torch.nn.Module = WeightedMAE
    validate: bool = True
    verbose: int = 2
