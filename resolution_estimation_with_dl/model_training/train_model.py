import argparse
import os
import torch
from torch.utils.data import DataLoader
from resolution_estimation_with_dl.model_training.model import UNet3D
from resolution_estimation_with_dl.model_training.training_utils import (
    initialize_weights,
    ResValDataset,
    Trainer,
)
from resolution_estimation_with_dl.model_training.training_config import TrainParameters


def set_up(output_dir):
    """Creates output directory for trained model weights if not exists"""
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


def train_model(
    train_data: str, valid_data: str, n_epochs: int, verbose: int
) -> UNet3D:
    """Train and (if necessary) validate model.

    Args:
        train_data (str): Path to the train data.
        valid_data (str): Path to the validation data.
        n_epochs (int): Number of iterations for model training.
        verbose (int): The detailing of the train process.

    Returns:
        UNet3D: Trained model.
    """
    trainset = ResValDataset(train_data)
    validset = ResValDataset(valid_data)
    trainloader = DataLoader(
        trainset, batch_size=TrainParameters.batch_size, shuffle=True, num_workers=2
    )
    validloader = DataLoader(
        validset, batch_size=TrainParameters.batch_size, shuffle=False, num_workers=2
    )
    DEVICE = TrainParameters.device
    model = UNet3D(
        upsample_mode=TrainParameters.upsample_mode,
        regularization=TrainParameters.regularization,
        align_corners=TrainParameters.align_corners,
    )

    if TrainParameters.initialize_weights:
        initialize_weights(model)

    optimizer = TrainParameters.optimizer(
        model.parameters(), lr=TrainParameters.learning_rate
    )
    criterion = TrainParameters.criterion(device=DEVICE)
    metric = TrainParameters.metric(device=DEVICE)
    trainer = Trainer(
        model,
        epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        device=DEVICE,
        metric=metric,
        validate=TrainParameters.validate,
        verbose=verbose,
    )
    try:
        trainer.fit()
    except KeyboardInterrupt:
        return model

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="resolution_estimation_with_dl/example_data/example_train_data.hdf5",
        help="Path to file with train samples.",
    )
    parser.add_argument(
        "--valid_data",
        default="resolution_estimation_with_dl/example_data/example_train_data.hdf5",
        help="Path to file with validation samples",
    )
    parser.add_argument(
        "--n_epochs",
        default=30,
        help="The number of epochs to train the model",
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=2,
        choices=[0, 1, 2],
        help="The higher the value, the more information about training will be displayed to the user",
        type=int,
    )
    parser.add_argument(
        "--out_weights_dir",
        default="resolution_estimation_with_dl/model_weights",
        help="The directory where model weights will be saved",
    )
    parser.add_argument(
        "--out_weights_filename",
        default="unet_3d_trained_weights.pth",
        help="The filename with trained model weights",
    )

    args = parser.parse_args()
    out_weights_dir = args.out_weights_dir

    set_up(out_weights_dir)

    # Model training
    model = train_model(
        train_data=args.train_data,
        valid_data=args.valid_data,
        n_epoches=args.n_epochs,
        verbose=args.verbose,
    )

    # Saving model weights
    torch.save(
        model.state_dict(), os.path.join(out_weights_dir, args.out_weights_filename)
    )


if __name__ == "__main__":
    main()
