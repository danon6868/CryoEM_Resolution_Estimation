import argparse
from model_training.model import *
from utils import run_model, save_mrc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_path",
        default="../model_weights/",
        help="Path to directory with saved models",
    )
    parser.add_argument(
        "--model_name",
        default="unet_3d_trained_dropout.pth",
        help="""Model name to load. `unet_3d_trained_dropout.pth` and `unet_3d_trained_batchnorm.pth` are available 
        and should be in `models_path` directory or they will be downloaded (it is about 400 Mb). If you want to use your own model, you 
        should put its weights into `models_path` directory yourself""",
    )
    parser.add_argument(
        "--electron_density_map",
        default="../example_data/14526_map.mrc",
        help="Electron density map for which you want to estimate local resolution map",
    )
    parser.add_argument(
        "--output_file_name",
        help="File name for local resolution map. By default it will be a `electron_density_map` name with `resulotion` suffix",
    )
    args = parser.parse_args()
    print(args.output_file_name)
