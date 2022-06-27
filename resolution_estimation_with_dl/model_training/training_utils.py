import h5py
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from loguru import logger


class ResValDataset(Dataset):
    """Pytorch Dataset that returns electron density map and local resolution map pairs by index"""

    def __init__(
        self,
        repository_file,
        map_transforms=Compose([torch.Tensor]),
        target_transforms=Compose([torch.Tensor]),
    ):
        self.repository = h5py.File(repository_file, mode="r")
        self.map_transforms = map_transforms
        self.target_transforms = target_transforms

    def __del__(self):
        self.repository.close()

    def __len__(self):
        return self.repository["map"].shape[0]

    def __getitem__(self, index):
        map_ = self.map_transforms(self.repository["map"][index])
        target_ = self.target_transforms(self.repository["target"][index])

        return map_[None, ...], target_[None, ...]


class WeightedMSELoss(nn.Module):
    """MSE loss with weights. We don't want to strongly penalize
    model for uncorrect predictions in area without a signal.
    For these points we will return constant (e.g. 100) (how it is done in Resmap)"""

    def __init__(self, outer_weights=0.0, reduction="mean", device="cpu"):
        super().__init__()
        self.outer_weights = outer_weights
        self.reduction = reduction
        self.device = device

    def forward(self, input, target):
        weights = torch.ones(input.shape).to(self.device)
        weights[target == 100.0] = self.outer_weights
        loss = weights * ((input - target) ** 2)

        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class WeightedMAE(nn.Module):
    """MAE metric with weights. We don't want to strongly penalize
    model for uncorrect predictions in area without a signal.
    For these points we will return constant (e.g. 100) (how it is done in Resmap)"""

    def __init__(self, outer_weights=0.0, reduction="mean", device="cpu"):
        super().__init__()
        self.outer_weights = outer_weights
        self.reduction = reduction
        self.device = device

    def forward(self, input, target):
        weights = torch.ones(input.shape).to(self.device)
        weights[target == 100.0] = self.outer_weights
        loss = weights * (input - target).abs()

        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class Trainer:
    """Class for model training and validation"""

    def __init__(
        self,
        model,
        epochs: int,
        criterion,
        optimizer,
        trainloader,
        validloader,
        device: str,
        metric,
        validate: bool = True,
        verbose: int = 2,
    ) -> None:
        """This class takes model, data, loss function, metric and other stuff for training
        and validation and train model during defined number of epoches.

        Args:
            model (_type_): _description_
            epochs (_type_): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
            trainloader (_type_): _description_
            validloader (_type_): _description_
            device (_type_): _description_
            metric (_type_): _description_
            validate (bool, optional): _description_. Defaults to True.
            verbose (int, optional): _description_. Defaults to 2.
        """

        self.model = model.to(device)
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device
        self.metric = metric
        self.validate = validate
        self.verbose = verbose
        self.scheduler1 = None
        self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3
        )

    def fit(self, epochs=None):
        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):

            train_loss, train_metric = self._train(self.trainloader)
            if self.validate:
                val_loss, val_metric = self._validate(self.validloader)
                if self.scheduler2 is not None:
                    self.scheduler2.step(val_loss)
            else:
                val_loss = "NO"

            if self.verbose > 0:
                logger_base_string = f"Epoch: {epoch+1}\tTrain loss: {train_loss:.6f}"
                logger_val_string = (
                    f"Validation loss: {val_loss:.6f}" if self.validate else ""
                )
                logger_train_metric = f"Train metric: {train_metric:.6f}"
                logger_val_metric = (
                    f"Validation metric: {val_metric:.6f}" if self.validate else ""
                )
                logger_complete_string = "\t".join(
                    [
                        logger_base_string,
                        logger_val_string,
                        logger_train_metric,
                        logger_val_metric,
                    ]
                )
                logger.info(logger_complete_string)

    def _train(self, loader):
        ## TODO define the way to use scheduler in init and do not hardcode it)
        self.model.train()
        epoch_loss = 0
        epoch_metric = 0
        for i, (maps, targets) in enumerate(loader):
            maps, targets = maps.to(self.device), targets.to(self.device)
            out = self.model(maps)
            loss = self.criterion(out, targets)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()

            if self.verbose > 1:
                print(f"\rTraining: batch {i+1} out of {len(loader)}", end="")

            self.optimizer.step()
            if self.scheduler1 is not None:
                self.scheduler1.step()

            epoch_metric += self.metric(out, targets).item()
            self._clear_vram(maps, targets, out)

        epoch_loss = epoch_loss / len(loader)
        epoch_metric = epoch_metric / len(loader)
        print("\n", end="")
        return epoch_loss, epoch_metric

    def _validate(self, loader):
        self.model.eval()
        epoch_loss = 0
        epoch_metric = 0
        with torch.no_grad():
            for i, (maps, targets) in enumerate(loader):
                maps, targets = maps.to(self.device), targets.to(self.device)
                out = self.model(maps)
                loss = self.criterion(out, targets)

                if self.verbose > 1:
                    print(f"\rValidation: batch {i+1} out of {len(loader)}", end="")

                epoch_loss += loss.item()
                epoch_metric += self.metric(out, targets).item()
                self._clear_vram(maps, targets, out)

        epoch_loss = epoch_loss / len(loader)
        epoch_metric = epoch_metric / len(loader)
        print("\n", end="")
        return epoch_loss, epoch_metric

    def _clear_vram(self, inputs, labels, outputs):
        inputs = inputs.to("cpu")
        labels = labels.to("cpu")
        outputs = outputs.to("cpu")
        del inputs, labels, outputs
        torch.cuda.empty_cache()


def initialize_weights(model):
    """Initialize model weights using He initialization (Kaiming uniform).

    Args:
        model (_type_): The instance of chosen model.
    """
    for part in model.children():
        try:
            for m in part:
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight.data, 1)
                    nn.init.constant_(m.bias.data, 0)
        except TypeError:
            continue
