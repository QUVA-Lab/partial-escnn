import argparse
import os
import sys
import time

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
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to PATH
sys.path.append(parent_dir)
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from data import MNIST_angle
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
from escnn2.r2convolution import R2Conv
from networks import CNN, SteerableCNN

from util import (
    get_kl_loss,
    get_shift_loss,
    get_irrepsmaps,
    number_of_params,
    calc_accuracy,
    log_learned_equivariance,
)


def make_data_loaders(config):
    mnist_test = MNIST_angle(train=False, max_val=9)

    mnist_train = MNIST_angle(max_val=9)

    test_loader = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=config.batch_size,
        num_workers=config.nr_workers,
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.nr_workers,
    )

    return train_loader, test_loader


def create_network(config):
    if config.group == "CNN":
        return CNN(mnist_type=config.mnist_type, n_classes=100).to(DEVICE)
    else:
        return SteerableCNN.from_group(
            config.group,
            mnist_type="single",
            restrict=config.restrict,
            n_classes=1 * config.angle + 10 * config.digit,
            basisexpansion="blocks" if not config.learn_eq else config.learn_eq,
            one_eq=config.one_eq,
            split=config.channel_splits,
            iteration=config.iteration,
        ).to(DEVICE)


def create_config(args, iteration):
    config = dict(
        iteration=iteration,
        group=args.group,
        lr=args.lr,
        epochs=args.epochs,
        optimizer="Adam",
        batch_size=args.batch_size,
        restrict=args.restrict,
        learn_eq_normalization=True if "norm" in f"{args.learn}" else False,
        learn_eq=args.learn,
        parametrization="fourier+act" if args.learn else "baseline",
        fourier_act="softmax" if args.learn else "-",
        L_in=args.L,
        L_out=args.L_out,
        kl_div=args.kl_div,
        alignment_loss=args.alignment_loss,
        activation="GatedNonLinearityUniform",
        mode_wandb=args.mode_wandb,
        one_eq=args.one_eq,
        channel_splits=args.channel_splits,
        nr_workers=args.nr_workers,
        angle=args.angle,
        digit=args.digit,
    )
    return config


def model_pipeline(config, mode_wandb):
    with wandb.init(
        project="thesis_mnist_angle",
        config=config,
        mode=mode_wandb,
        tags=[str(DEVICE)],
        reinit=True,
    ):
        config = wandb.config
        model = create_network(config)
        train_loader, test_loader = make_data_loaders(config)
        wandb.config["network_name"] = model.network_name
        wandb.config["nr_params"] = number_of_params(model)
        print(f'Nr of model params: {wandb.config["nr_params"]}')
        config = wandb.config
        wandb.run.name = f"{model.network_name} {'c' if config.channel_splits else ''} {'one_eq' if config.one_eq else ''}"
        wandb.config["run_name"] = wandb.run.name
        print(wandb.run.name)

        train_nn(model, train_loader, test_loader, config)


def train_nn(model, train_loader, test_loader, config):
    use_amp = USE_AMP
    epochs = config.epochs
    model = model.to(DEVICE)
    model.train()
    loss_digit_fn = torch.nn.CrossEntropyLoss()
    loss_angle_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.watch(
        model,
        [loss_digit_fn, loss_angle_fn],
        log="all",
        log_freq=len(train_loader),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    irrepmaps = get_irrepsmaps(model)

    log_learned_equivariance(irrepmaps, 0, config)

    max_test_acc, min_angle_loss = 0, np.inf

    for epoch in range(epochs):
        start_time = time.time()
        (
            running_total_loss,
            running_digit_loss,
            running_angle_loss,
            running_digit_acc,
            running_shift_loss,
            running_kl_loss,
        ) = (0, 0, 0, 0, 0, 0)

        for i, (x, target_digit, target_angle) in enumerate(train_loader):
            start = time.time()
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                x = x.to(DEVICE)
                target_digit, target_angle = target_digit.to(DEVICE), target_angle.to(
                    DEVICE
                )
                preds = model(x)
                kl_loss = get_kl_loss(irrepmaps)
                running_kl_loss += kl_loss
                shift_loss = get_shift_loss(irrepmaps)
                running_shift_loss += shift_loss

                loss = config.alignment_loss * shift_loss + config.kl_div * kl_loss
                if config.digit:
                    running_digit_acc += calc_accuracy(preds[:, :10], target_digit)
                    loss_digit = loss_digit_fn(preds[:, :10], target_digit)
                    loss += loss_digit
                    running_digit_loss += loss_digit.item()
                    running_total_loss += loss_digit.item()
                if config.angle:
                    loss_angle = loss_angle_fn(
                        torch.cos(preds[:, -1]), torch.cos(target_angle)
                    )
                    loss += loss_angle
                    running_angle_loss += loss_angle.item()
                    running_total_loss += loss_angle.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            end = time.time()
            # print(end - start, len(train_loader))

        print("########################################################")
        print(
            f"Running epoch loss: {running_total_loss/(i+1):.4f}, acc: {running_digit_acc/(i+1):.4f}, angle_loss {running_angle_loss/(i+1):.4f}"
        )
        print(f"Epoch time: {time.time() - start_time :.4f}")

        test_loss, test_angle_loss, test_acc, test_digit_loss = test_nn(
            model, test_loader, loss_digit_fn, loss_angle_fn, config
        )

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            min_angle_loss = test_angle_loss

        print(
            f"Epoch {epoch}: test loss {test_loss:.4f} test acc:{test_acc:.4f} digit loss: {test_digit_loss:.4f} angle loss: {test_angle_loss:.4f}"
        )
        print("\n########################################################")
        wandb.log(
            {
                "Train Total Loss": running_total_loss / (i + 1),
                "Train angle loss": running_angle_loss / (i + 1),
                "Train digit loss": running_digit_loss / (i + 1),
                "Train Accuracy": running_digit_acc / (i + 1),
                "Alignment Loss": running_shift_loss / (i + 1),
                "KL-Divergence": running_kl_loss / (i + 1),
                "Test Total  Loss": test_loss,
                "Test digit Loss": test_digit_loss,
                "Test angle Loss": test_angle_loss,
                "Test digit acc": test_acc,
                "min angle loss": min_angle_loss,
                "max Test Accuracy": max_test_acc,
            },
            step=epoch + 1,
        )

        log_learned_equivariance(irrepmaps, epoch + 1, config)

    if config.iteration == 0:
        log_equivariance_error(model, config, test_loader)
    save_and_log_model(dict(config), model, optimizer)


def test_nn(model, test_loader, loss_digit_fn, loss_angle_fn, config):
    use_amp = USE_AMP
    model.eval()
    loss, total_loss_angle, total_loss_digit, acc = 0, 0, 0, 0
    with torch.no_grad():
        for i, (x, target_digit, target_angle) in enumerate(test_loader):
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                x = x.to(DEVICE)
                target_digit, target_angle = target_digit.to(DEVICE), target_angle.to(
                    DEVICE
                )

                preds = model(x)
                loss = 0
                if config.digit:
                    acc += calc_accuracy(preds[:, :10], target_digit)
                    loss_digit = loss_digit_fn(preds[:, :10], target_digit).item()
                    total_loss_digit += loss_digit
                    loss += loss_digit
                if config.angle:
                    loss_angle = loss_angle_fn(
                        torch.cos(preds[:, -1]), torch.cos(target_angle)
                    ).item()
                    total_loss_angle += loss_angle
                    loss += loss_angle

    model.train()

    return (
        loss / len(test_loader),
        total_loss_angle / len(test_loader),
        acc / len(test_loader),
        total_loss_digit / len(test_loader),
    )


def log_equivariance_error(model, config, loader):
    model.eval()
    layers = list(filter(lambda x: isinstance(x, R2Conv), model.modules()))
    if not layers:
        return
    layer_ids = [layer.layer_id for layer in layers]
    layer_offset = min(layer_ids)
    eq_modules = [
        [
            [layer for layer in module]
            if isinstance(module, nn.SequentialModule)
            else [module]
            for module in layers
        ]
        for layers in model.layers_eq
    ]
    eq_modules = [
        [module for sublist in eq_channel_modules for module in sublist]
        for eq_channel_modules in eq_modules
    ]

    with torch.no_grad():
        for channel_id, channel_modules in enumerate(eq_modules):
            data, *_ = next(iter(loader))
            x = data[:10]
            layer_num = 0
            for layer in channel_modules:
                in_type = layer.in_type
                x = in_type(x).to(DEVICE)
                preds = layer(x)
                x_shape = list(x.shape)[2]

                if isinstance(layer, R2Conv):
                    group = in_type.fibergroup
                    if group.continuous:
                        n = 50
                    else:
                        try:
                            n = group.rotation_order
                        except:
                            n = group.N
                    try:
                        testing_elements = [
                            elem for elem in group.testing_elements(n=n)
                        ]
                    except TypeError:
                        testing_elements = [elem for elem in group.testing_elements()]
                    factor = len(testing_elements) // n
                    g_elements = np.linspace(
                        0, 2 * np.pi * factor, len(testing_elements)
                    )
                    g_elements = np.concatenate(
                        [
                            np.linspace(
                                i * (2 * np.pi), 2 * np.pi * (i + 1), n, endpoint=False
                            )
                            for i in range(factor)
                        ]
                    )

                    if x_shape > 1 and x_shape < 32:
                        x = in_type(
                            interpolate(
                                x.tensor,
                                size=(32, 32),
                                mode="bilinear",
                                align_corners=True,
                            )
                        )
                    layer_id = layer.layer_id

                    out_type = layer.out_type
                    start_channel = 0
                    errors = []
                    # preds_norm = torch.empty_like(preds.tensor)
                    # for channel in out_type:
                    #     channel_norm = torch.linalg.norm(
                    #         preds.tensor[
                    #             :, start_channel : start_channel + channel.size, :, :
                    #         ],
                    #         axis=1,
                    #         keepdim=True,
                    #     )
                    #     preds_norm[
                    #         :, start_channel : start_channel + channel.size, :, :
                    #     ] = channel_norm
                    #     start_channel += channel.size

                    for t in testing_elements:
                        augmentend_data = x.transform(t)
                        if x_shape > 1 and x_shape < 32:
                            augmentend_data = in_type(
                                interpolate(
                                    augmentend_data.tensor,
                                    size=(x_shape, x_shape),
                                    mode="bilinear",
                                    align_corners=True,
                                )
                            )
                        aug_preds = layer(augmentend_data).tensor

                        channel_errors = []
                        start_channel = 0
                        preds_transform = preds.transform(t).tensor
                        for channel in out_type:
                            aug_preds_channel = aug_preds[
                                :, start_channel : start_channel + channel.size, :, :
                            ]
                            preds_transform_channel = preds_transform[
                                :, start_channel : start_channel + channel.size, :, :
                            ]

                            channel_errors.append(
                                (
                                    torch.linalg.norm(
                                        aug_preds_channel - preds_transform_channel,
                                        axis=1,
                                        keepdim=True,
                                    )
                                )
                                .mean()
                                .cpu()
                                .numpy()
                            )

                        error = np.mean(channel_errors)

                        errors.append(error.item())
                    data = [
                        [
                            g_element,
                            error,
                            layer_id,
                            config.run_name,
                            f"{config.run_name}{config.iteration}",
                        ]
                        for (g_element, error) in zip(g_elements, errors)
                    ]

                    table = wandb.Table(
                        data=data,
                        columns=[
                            "transformation element",
                            "error",
                            "layer_id",
                            "run_name",
                            "run_name_iter",
                        ],
                    )

                    wandb.log(
                        {
                            f"Equivariance error layer {layer_num} channel {channel_id} iter {config.iteration}": wandb.plot.line(
                                table,
                                "transformation element",
                                "error",
                                title=f"Equivariance layer {layer_num}/{layer_id - layer_offset} error channel {channel_id}",
                            )
                        }
                    )

                    plt.title(
                        f"Equivariance layer {layer_num} error channel {channel_id}"
                    )

                    plt.plot(g_elements, errors)
                    plt.ylim([0, 1.1 * max(errors) + 1e-4])
                    plt.ylabel("Equivariance Error")
                    plt.xlabel("Groupelement g")
                    wandb.log(
                        {
                            f"Equivariance error layer {layer_num}/{layer_id - layer_offset} channel {channel_id} iter {config.iteration}": wandb.Image(
                                plt
                            )
                        }
                    )
                    plt.close()
                    layer_num += 1
                x = preds.tensor
    model.train()


def save_and_log_model(config, model, optimizer):
    try:
        os.makedirs("checkpoints/", exist_ok=True)
    except FileNotFoundError:
        pass
    except FileExistsError:
        pass

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        f"checkpoints/mnist_angle/{config['run_name']} iteration {config['iteration']}.pth",
    )
    try:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(
            f"checkpoints/mnist_angle/{config['run_name']} iteration {config['iteration']}.pth"
        )
        wandb.run.log_artifact(artifact)
    except:
        pass


if __name__ == "__main__":
    DEVICE = "cuda"
    USE_AMP = False
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--group",
        type=str,
        help="Transformation group network should be equivariant to. Options are C/D{0, 1, 2, 4, 8, 12}, SO2, O2, trivial Steerable. D1=D0 and C1=C0='trivial'. CNN can also be chosen, in which case a regular CNN is used",
        default="SO2",
    )

    parser.add_argument(
        "-m",
        "--mode_wandb",
        type=str,
        help="mode of wandb: online, offline, disabled",
        default="online",
    )

    parser.add_argument(
        "-r",
        "--restrict",
        action="store_true",
        help="restricts the network to trival in the last two layers",
    )

    parser.add_argument("--epochs", type=int, default=25, help="nr of epochs to run")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--batch_size", type=int, default=256)

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

    parser.add_argument(
        "-kl",
        "--kl_div",
        type=float,
        default=3,
    )

    parser.add_argument(
        "-align",
        "--alignment_loss",
        type=float,
        default=5,
    )

    parser.add_argument("--one_eq", action="store_true")

    parser.add_argument(
        "-c",
        "--channel_splits",
        type=str,
        default=None,
    )

    parser.add_argument("-a", "--angle", action="store_true")

    parser.add_argument("-d", "--digit", action="store_true")

    args = parser.parse_args()

    assert args.angle or args.digit

    for iteration in args.iterations:
        config = create_config(args, iteration)
        model_pipeline(config, args.mode_wandb)
