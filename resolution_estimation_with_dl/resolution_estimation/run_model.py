import argparse
import os
import gdown
from pathlib import Path
from loguru import logger
from ..model_training.model import UNet3D
from ..model_training.training_config import TrainParameters
from ..resolution_estimation.utils import run_model, save_mrc


OUTPUT_DIR = "results/"


def set_up():
    """Creates output directory for local resolution map if not exists"""
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


def download_weights(model_name):
    if model_name == "unet_3d_trained_dropout.pth":
        url = "https://drive.google.com/u/0/uc?id=1-Imt4lolu-dMMsNItYgLV8GFDKDGUCr0&export=download"
        out_file_name = os.path.join(
            "resolution_estimation_with_dl",
            "model_weights",
            "unet_3d_trained_dropout.pth",
        )
        logger.info(f"Downloading {model_name} in {out_file_name}...")
        gdown.download(url, out_file_name, quiet=False)
        logger.info(
            f"Weights were successfully downloaded. Now you can predict your local resolution map!"
        )

    elif model_name == "unet_3d_trained_batchnorm.pth":
        url = "https://drive.google.com/u/0/uc?id=1XGV2X9DHW3JqTXuyi4OBGCwrQrtfcOfe&export=download"
        out_file_name = os.path.join(
            "resolution_estimation_with_dl",
            "model_weights",
            "unet_3d_trained_batchnorm.pth",
        )
        logger.info(f"Downloading {model_name} in {out_file_name}...")
        gdown.download(url, out_file_name, quiet=False)
        logger.info(
            f"Weights were successfully downloaded. Now you can predict your local resolution map!"
        )
    else:
        raise NameError(
            f"{model_name} is not valid. Use `-h` option to find out valid models."
        )


if __name__ == "__main__":
    # Parse agruments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_path",
        default="resolution_estimation_with_dl/model_weights/",
        help="Path to directory with saved models",
    )
    parser.add_argument(
        "--model_name",
        default="unet_3d_trained_dropout.pth",
        choices=["unet_3d_trained_dropout.pth", "unet_3d_trained_batchnorm.pth"],
        help="""Model name to load. `unet_3d_trained_dropout.pth` and `unet_3d_trained_batchnorm.pth` are available 
        and should be in `models_path` directory or they will be downloaded (it is about 400 Mb). If you want to use your own model, you 
        should put its weights into `models_path` directory yourself. Experiments shown that model with dropout was better on our experimental data""",
    )
    parser.add_argument(
        "--electron_density_map",
        default="resolution_estimation_with_dl/example_data/14526_map.mrc",
        help="Electron density map for which you want to estimate local resolution map. By default it will be example map",
    )
    parser.add_argument(
        "--output_file_name",
        help="File name for local resolution map. By default it will be a `electron_density_map` name with `resulotion` suffix",
    )
    args = parser.parse_args()

    # Set up directories and download model weights if it is required
    set_up()
    electron_density_file_name = Path(args.electron_density_map).name

    if args.output_file_name is None:
        local_resolution_file_name = f"{electron_density_file_name[:-4]}_resolution.mrc"
    else:
        local_resolution_file_name = args.output_file_name

    if args.model_name not in os.listdir(
        os.path.join("resolution_estimation_with_dl", "model_weights/")
    ):
        download_weights(args.model_name)
    else:
        logger.info(
            f"Weights of '{args.model_name}' are available on you computer already. Now you can predict your local resolution map!"
        )
    path_to_weights = os.path.join(
        "resolution_estimation_with_dl", "model_weights", args.model_name
    )

    # Set up model
    unet_3d = UNet3D(
        upsample_mode=TrainParameters.upsample_mode,
        regularization=TrainParameters.regularization,
        align_corners=TrainParameters.align_corners,
    )
    logger.info(f"Start local resolution map estimation using UNet3D...")
    padded_map, local_resolution_map = run_model(
        args.electron_density_map, TrainParameters.device, unet_3d, path_to_weights
    )
    logger.info(
        f"Saving final electron density and local resolution maps into {OUTPUT_DIR}..."
    )
    save_mrc(padded_map, os.path.join(OUTPUT_DIR, electron_density_file_name))
    save_mrc(local_resolution_map, os.path.join(OUTPUT_DIR, local_resolution_file_name))
    logger.info(
        f"Final electron density and local resolution maps were saved successfully!"
    )
