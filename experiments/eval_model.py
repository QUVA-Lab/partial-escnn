import argparse
import os
import sys
import time
import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import wandb

plt.rcParams["figure.figsize"] = (15, 10)

import torchvision.transforms.functional as TF
from escnn import group, gspaces, nn
from sklearn.metrics import confusion_matrix

sys.path.append("..")
print(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to PATH
sys.path.append(parent_dir)
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from data import MNIST_Double
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
from escnn2.r2convolution import R2Conv
from networks import (
    CNN,
    SteerableCNN,
    CNN3D,
    SteerableCNN3D,
    SteerableCNN3DResnet,
    CNN3DResnet,
)

from util import (
    Rotate90Transform,
    get_kl_loss,
    get_shift_loss,
    get_irrepsmaps,
    number_of_params,
    calc_accuracy,
    log_learned_equivariance,
)


def create_config(args, iteration):
    config = dict(
        iteration=iteration,
        mnist_type=args.mnist_type,
        restrict=args.restrict,
        group=args.group,
        learn_eq=args.learn,
        fourier_act="softmax" if args.learn else "-",
        L_in=args.L,
        L_out=args.L_out,
        activation="GatedNonLinearityUniform",
        one_eq=args.one_eq,
        channels=args.channels,
        dataset=args.dataset,
        n_classes=2,
        n_channels=1,
        resnet=args.resnet,
    )
    return config


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_network(config):
    config = AttrDict(config)
    if "3d" not in config.dataset:
        if config.group == "CNN":
            cnn = CNN
            return cnn(
                mnist_type=config.mnist_type,
                n_classes=config.n_classes,
                n_channels=config.n_channels,
            ).to(DEVICE)
        else:
            steerable = SteerableCNN
            return steerable.from_group(
                config.group,
                mnist_type=config.mnist_type,
                restrict=config.restrict,
                n_classes=config.n_classes,
                n_channels=config.n_channels,
                basisexpansion="blocks" if not config.learn_eq else config.learn_eq,
                one_eq=config.one_eq,
                # channels=config.channels,
                iteration=config.iteration,
                L_in=config.L_in,
                L_out=config.L_out,
            ).to(DEVICE)
    else:
        if config.group == "CNN":
            cnn = CNN3DResnet if config.resnet else CNN3D
            return cnn(
                mnist_type=config.mnist_type,
                n_classes=config.n_classes,
                n_channels=config.n_channels,
                c=config.channels,
            ).to(DEVICE)
        else:
            steerable = SteerableCNN3DResnet if config.resnet else SteerableCNN3D
            return steerable.from_group(
                config.group,
                mnist_type=config.mnist_type,
                restrict=config.restrict,
                n_classes=config.n_classes,
                n_channels=config.n_channels,
                basisexpansion="blocks" if not config.learn_eq else config.learn_eq,
                one_eq=config.one_eq,
                channels=config.channels,
                iteration=config.iteration,
                L_in=config.L_in,
                L_out=config.L_out,
            ).to(DEVICE)


if __name__ == "__main__":
    DEVICE = "cuda"
    USE_AMP = True
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--group",
        type=str,
        help="Transformation group network should be equivariant to. Options are C/D{0, 1, 2, 4, 8, 12}, SO2, O2, trivial Steerable. D1=D0 and C1=C0='trivial'. CNN can also be chosen, in which case a regular CNN is used",
        default="SO2",
    )

    parser.add_argument(
        "-r",
        "--restrict",
        action="store_true",
        help="restricts the network to trival in the last two layers",
    )

    parser.add_argument(
        "--mnist_type", type=str, choices=["single", "double"], default="single"
    )

    parser.add_argument("--nr_workers", type=int, default=12)

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        nargs="*",
        help="Number of iterations to run",
        default=[0],
    )

    parser.add_argument(
        "--learn", type=str, choices=["learn_eq", "learn_eq_norm"], default=False
    )

    parser.add_argument(
        "-L",
        type=int,
        default=2,
    )

    parser.add_argument(
        "-L_out",
        type=int,
        default=4,
    )

    parser.add_argument("--one_eq", action="store_true")

    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "pathmnist",
            "organamnist",
            "organcmnist",
            "organsmnist",
            "breastmnist",
            "retinamnist",
            "dermamnist",
            "organmnist3d",
            "nodulemnist3d",
            "fracturemnist3d",
            "adrenalmnist3d",
            "vesselmnist3d",
            "synapsemnist3d",
        ],
        default="organcmnist",
    )

    parser.add_argument("--resnet", action="store_true")

    args = parser.parse_args()
    for iteration in args.iterations:
        config = create_config(args, iteration)
        model = create_network(config).to("cuda")
        print(number_of_params(model))

        api = wandb.Api()
        artifact = api.artifact(
            "lveefkind/thesis_med_mnist_3d_test_datasets/model:v507", type="model"
        )
        artifact_dir = artifact.download()
        complete_location = artifact.file(artifact_dir)
        thing = torch.load(complete_location)
        model.train()
        x = torch.randn(2, 1, 28, 28, 28).to("cuda")
        model(x)

        # for var in thing["model_state_dict"]:
        #     if not var in model.state_dict():
        #         print(var)

        model.load_state_dict(thing["model_state_dict"], strict=False)
        x = torch.randn(2, 1, 28, 28, 28).to("cuda")
        model(x)
        irrepmaps = get_irrepsmaps(model)
        print(get_kl_loss(irrepmaps))
