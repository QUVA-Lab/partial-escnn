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
script_dir = os.path.dirname(
    os.path.abspath("/home/lars/Studie/lars_thesis/experiments/eval_model.py")
)
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
from networks import CNN, SteerableCNN, CNN3D, SteerableCNN3D, SteerableCNN3DResnet
from util import (
    Rotate90Transform,
    get_kl_loss,
    get_shift_loss,
    get_irrepsmaps,
    number_of_params,
    calc_accuracy,
    log_learned_equivariance,
    plot_signal,
)
import escnn

# %load_ext autoreload
# %autoreload 2


def make_data_loaders(config):
    data_flag = config["dataset"]
    download = True
    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    if not "3d" in config["dataset"]:
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
    else:
        data_transform = data_transform = transforms.Compose(
            [
                lambda x: torch.FloatTensor(x),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    # data_transform = lambda x: x

    train_dataset = DataClass(
        split="train", transform=data_transform, download=download
    )
    test_dataset = DataClass(split="test", transform=data_transform, download=download)

    val_dataset = DataClass(split="val", transform=data_transform, download=download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=12,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=48 if "3d" in config["dataset"] else 1024,
        shuffle=False,
        num_workers=12,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=48 if "3d" in config["dataset"] else 1024,
        shuffle=False,
        num_workers=12,
    )

    return train_loader, test_loader, val_loader


def create_config(args, iteration):
    config = dict(
        iteration=0,
        mnist_type="single",
        restrict=False,
        group="O3",
        learn_eq="learn_eq_norm",
        fourier_act="softmax",
        L_in=2,
        L_out=2,
        activation="GatedNonLinearityUniform",
        one_eq=False,
        channels=6,
        dataset="organmnist3d",
        n_classes=11,
        n_channels=1,
    )
    return config


def create_network(config):
    if config["group"] == "CNN":
        cnn = CNN3D if "3d" in config["dataset"] else CNN
        return cnn(
            mnist_type=config.mnist_type,
            n_classes=config.n_classes,
            n_channels=config.n_channels,
            c=config.channels,
        ).to(DEVICE)
    else:
        steerable = SteerableCNN3DResnet if "3d" in config["dataset"] else SteerableCNN
        return steerable.from_group(
            config["group"],
            mnist_type=config["mnist_type"],
            restrict=config["restrict"],
            n_classes=config["n_classes"],
            n_channels=config["n_channels"],
            basisexpansion="blocks" if not config["learn_eq"] else config["learn_eq"],
            one_eq=config["one_eq"],
            channels=config["channels"],
            iteration=config["iteration"],
            L_in=config["L_in"],
            L_out=config["L_out"],
        ).to(DEVICE)


DEVICE = "cuda"
USE_AMP = True
args = "1"
iteration = 0
config = create_config(args, iteration)
model = create_network(config)
x = torch.randn(2, 1, 28, 28, 28).to(DEVICE)
model(x)

irrepmaps = get_irrepsmaps(model)
print(get_kl_loss(irrepmaps))

_, loader, _ = make_data_loaders(config)

import escnn2


api = wandb.Api()
artifact = api.artifact("lveefkind/med_3d_final/model:v45", type="model")
artifact_dir = artifact.download()
complete_location = artifact.file(artifact_dir)
thing = torch.load(complete_location)
model.eval()
model.load_state_dict(thing["model_state_dict"], strict=False)
model = model.to("cuda")
irrepmaps = get_irrepsmaps(model)
# for irrepmap in irrepmaps:
#     escnn2.kernels.irreps_mapping.set_grad_flag.recompute_ft[irrepmap] = True
print(escnn2.kernels.irreps_mapping.set_grad_flag.recompute_ft)
x = torch.randn(2, 1, 28, 28, 28).to("cuda")
model(x)

# print(irrepmaps[5].fts_in)
# print(irrepmaps[5].fts_out)


irrepmaps = get_irrepsmaps(model)
get_kl_loss(irrepmaps)

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from medmnist import OrganMNIST3D


def log_equivariance_error2(model, loader, irrepmaps):
    dataset = OrganMNIST3D("train")
    labels_names = dataset.info["label"]
    # KEEP n even!!
    _, _, data = irrepmaps[2].get_distribution(n=38, sphere=True)
    # print(data['f'][0])
    # figs = plot_signal(data)
    # figs[0].show()
    # return
    model.eval()
    eq_layers = [seq for seq in model.layers_eq]
    labels_to_take = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])[:5]

    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
            enabled=False,
        ):
            loader = iter(loader)
            count = 0
            n_classes_total = 11
            n_classes = len(labels_to_take)
            imgs_list = []
            while count != 1:
                nr_unique = 0
                while nr_unique != n_classes_total:
                    imgs, labels = next(loader)
                    nr_unique = len(np.unique(labels))
                unique = np.unique(labels, return_index=True)[1][labels_to_take]
                # unique = unique[0:2]
                imgs = imgs[unique]
                labels = labels[unique].to(DEVICE)
                imgs = model.in_type(imgs).to(DEVICE)
                preds = model(imgs.tensor)
                acc = (
                    torch.argmax(preds, axis=1) == labels.flatten()
                ).sum() / n_classes
                if acc == 1:
                    count += 1
                    imgs_list += [imgs.tensor]

            imgs, labels = 0, 0
            x = model.in_type(torch.cat(imgs_list, axis=0))
            preds = model(x.tensor)
            norm = torch.linalg.norm(preds, axis=1, keepdim=True)
            f_id = torch.zeros((x.shape[0], len(data["f_id"])), device=preds.device)
            for i in tqdm(range(len(data["grid_id"]))):
                elem = data["grid_id"][i]
                x_transform = x.transform(elem)
                preds_transform = model(x_transform.tensor)
                f_id[:, i] = (
                    torch.linalg.norm(preds_transform - preds, axis=1, keepdim=True)
                    / norm
                ).view(-1)
            f_id = f_id.cpu().numpy()

            f_id = f_id.reshape(f_id.shape[0] // n_classes, n_classes, -1)
            f_id = np.mean(f_id, axis=0)

            f = torch.zeros((x.shape[0], len(data["f"])), device=preds.device)
            for i in tqdm(range(len(data["grid"]))):
                elem = data["grid"][i]
                x_transform = x.transform(elem)
                preds_transform = model(x_transform.tensor)
                f[:, i] = (
                    torch.linalg.norm(preds_transform - preds, axis=1, keepdim=True)
                    / norm
                ).view(-1)
            f = f.cpu().numpy()

            f = f.reshape(f.shape[0] // n_classes, n_classes, -1)
            f = np.mean(f, axis=0)

            # Fix keyerror due to multiple thingiies
            max_val = max(np.max(f), np.max(f_id))
            for label in range(f.shape[0]):
                print(labels_names[str(label)])
                f[label] = 1 - (f[label] / max_val)
                f_id[label] = 1 - (f_id[label] / max_val)

            for label in range(f.shape[0]):
                data_2 = data
                data_2["f"] = f[label]
                data_2["f_id"] = f_id[label]
                figs = plot_signal(data_2, f_max=np.max(f), f_min=np.min(f))
                for i, fig in enumerate(figs):
                    fig.write_html(
                        f'organ_{labels_names[str(label)]}{"_refl" if i else ""}.html'
                    )


# Entire model try 2
log_equivariance_error2(model, loader, irrepmaps)
