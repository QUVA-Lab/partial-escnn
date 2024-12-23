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

from sklearn.model_selection import train_test_split

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
import random

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
    SteerableRPP3DResnet,
)

# torch.use_deterministic_algorithms(True)

from util import (
    Rotate90Transform,
    get_kl_loss,
    get_shift_loss,
    get_irrepsmaps,
    number_of_params,
    calc_accuracy,
    log_learned_equivariance,
    set_seed,
    RPPConv_L2,
)

# print(torch.backends.opt_einsum.enabled)
# print(torch.backends.opt_einsum.enabled)


def make_data_loaders(config):
    data_flag = config.dataset
    download = True
    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    config.n_classes = n_classes

    config.n_channels = n_channels

    config.labels = info["label"]

    DataClass = getattr(medmnist, info["python_class"])

    if not "3d" in config.dataset:
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
    counts = np.unique(train_dataset.labels, return_counts=True)[1]
    ratio = config.ratio
    if ratio < 1:
        train_idx, _ = train_test_split(
            range(len(train_dataset)),
            test_size=1 - ratio,
            stratify=train_dataset.labels,
            random_state=42,
        )

        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    test_dataset = DataClass(split="test", transform=data_transform, download=download)
    val_dataset = DataClass(split="val", transform=data_transform, download=download)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.iteration)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.nr_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=48 if "3d" in config.dataset else 1024,
        shuffle=False,
        num_workers=config.nr_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=48 if "3d" in config.dataset else 1024,
        shuffle=False,
        num_workers=config.nr_workers,
    )

    return train_loader, test_loader, val_loader, counts


def create_network(config):
    if config.activation == "gated":
        activation = nn.GatedNonLinearityUniform
    elif config.activation == "QFourier":
        activation = nn.QuotientFourierELU
    elif config.activation == "Fourier":
        activation = nn.FourierELU

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
                learn_eq=config.learn_eq,
                normalise_basis=config.normalization,
                one_eq=config.one_eq,
                iteration=config.iteration,
                L_in=config.L_in,
                L_out=config.L_out,
                activation=activation,
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
        elif config.rpp:
            assert config.resnet
            steerable = SteerableRPP3DResnet
        else:
            steerable = SteerableCNN3DResnet if config.resnet else SteerableCNN3D
        return steerable.from_group(
            config.group,
            mnist_type=config.mnist_type,
            restrict=config.restrict,
            n_classes=config.n_classes,
            n_channels=config.n_channels,
            learn_eq=config.learn_eq,
            normalise_basis=config.normalization,
            one_eq=config.one_eq,
            channels=config.channels,
            iteration=config.iteration,
            L_in=config.L_in,
            L_out=config.L_out,
            activation=activation,
            invariant=config.invariant,
        ).to(DEVICE)


def train_nn(model, train_loader, test_loader, val_loader, config, counts=None):
    use_amp = USE_AMP
    epochs = config.epochs
    model = model.to(DEVICE)
    model.train()
    weights = torch.FloatTensor(1 / counts).to(DEVICE) if counts is not None else None
    weights = None

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    irrepmaps = get_irrepsmaps(model)

    if config.rpp:
        regularizer = RPPConv_L2
    else:
        regularizer = None

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    wandb.watch(model, loss_fn, log="all", log_freq=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if not "3d" in config.dataset:
        log_learned_equivariance(
            irrepmaps,
            0,
            config,
            sphere=True if not "3d" in config.dataset else False,
        )
    # log_learned_equivariance(
    #     irrepmaps,
    #     0,
    #     config,
    #     sphere=True,
    # )

    max_test_acc, max_val_acc, min_val_loss = 0, 0, np.inf

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        (
            running_ep_loss,
            running_acc,
            running_shift_loss,
            running_kl_loss,
            running_kl_uni,
        ) = (0, 0, 0, 0, 0)
        for i, (x, targets) in enumerate(train_loader):
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                start = time.time()
                x = x.to(DEVICE)
                targets = targets.to(DEVICE).squeeze()
                preds = model(x)
                running_acc += calc_accuracy(preds, targets)
                loss = loss_fn(preds, targets)
                running_ep_loss += loss.item()
                kl_loss, kl_uniform = get_kl_loss(irrepmaps)
                running_kl_loss += kl_loss
                running_kl_uni += kl_uniform
                shift_loss = get_shift_loss(irrepmaps)
                running_shift_loss += shift_loss

                # if kl_loss == 0:
                loss += (
                    config.alignment_loss * shift_loss
                    + config.kl_div * kl_loss
                    + config.kl_uniform * kl_uniform
                )

            if regularizer is not None:
                reg = regularizer(model, config.conv_wd, config.basic_wd)
                loss += reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # for irrepmap in irrepmaps:
            #     irrepmap.fts_in.requires_grad = False
            end = time.time()

        if epoch == 9 and config.early_stop:
            for irrepmap in irrepmaps:
                irrepmap.fts_in.requires_grad = False

        print("########################################################")
        print(
            f"Running epoch loss: {running_ep_loss/(i+1):.4f}, running acc: {running_acc/(i+1):.4f}"
        )
        print(f"Epoch time: {time.time() - start_time :.4f}")

        test_loss, test_acc = test_nn(model, test_loader, loss_fn, "test")
        val_loss, val_acc = test_nn(model, val_loader, loss_fn, "val")

        if val_acc > max_val_acc or (
            val_acc == max_val_acc and val_loss < min_val_loss
        ):
            print("Model Improvement!")
            min_val_loss = val_loss
            max_test_acc = test_acc
            max_val_acc = val_acc
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                model_name = log_checkpoint(dict(config), model, optimizer)

        print(
            f"Epoch {epoch}: test loss {test_loss:.4f} test acc:{test_acc:.4f} val loss {val_loss:.4f} val acc:{val_acc:.4f}"
        )
        print("\n########################################################")
        wandb.log(
            {
                "Train Loss": running_ep_loss / (i + 1),
                "Running Shift loss": running_shift_loss / (i + 1),
                "Shift loss": shift_loss,
                "Running KL-Divergence": running_kl_loss / (i + 1),
                "KL-Divergence": kl_loss,
                "Running KL-Divergence_uniform": running_kl_uni / (i + 1),
                "KL-Divergence-uniform": kl_uniform,
                "Train Accuracy": running_acc / (i + 1),
                "Test Loss": test_loss,
                "val_loss": val_loss,
                "Test Accuracy": test_acc,
                "Val Accuracy": val_acc,
                "max Test Accuracy": max_test_acc,
            },
            step=epoch + 1,
        )
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
            enabled=use_amp,
        ):
            model.eval()
            if (epoch % 10 == 0 or epoch + 1 == epochs) and config.iteration != 9:
                log_learned_equivariance(
                    irrepmaps,
                    epoch + 1,
                    config,
                    sphere=False,
                )
                log_confusion_matrix(model, test_loader, config)

    if "organ" not in config.dataset:
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
            enabled=use_amp,
        ):
            model.train()
            model = load_checkpoint(model, model_name)
    # log_learned_equivariance(
    #     irrepmaps,
    #     epoch + 2,
    #     config,
    #     sphere=True,
    # )
    with torch.autocast(
        device_type=DEVICE,
        dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
        enabled=use_amp,
    ):
        log_learned_equivariance(
            irrepmaps,
            epoch + 1,
            config,
            sphere=True if config.iteration == 0 else False,
        )
        log_confusion_matrix(model, test_loader, config)
        if config.iteration == 0 and not "3d" in config.dataset:
            log_equivariance_error(model, config, test_loader)
    save_and_log_model(dict(config), model, optimizer)


def load_checkpoint(model, model_name):
    checkpoint = torch.load(f"model_checkpoints/med_mnist/{model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
    # model.load_state_dict


def log_checkpoint(config, model, optimizer):
    model.train()
    try:
        os.makedirs("model_checkpoints/med_mnist", exist_ok=True)
    except FileNotFoundError:
        pass
    except FileExistsError:
        pass
    model_name = f"{config['dataset']}_{config['run_name']} iteration {config['iteration']}_TMP_{wandb.run.id}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        f"model_checkpoints/med_mnist/{model_name}",
    )
    model.train()
    return model_name


def log_equivariance_error(model, config, loader):
    model.eval()
    layers = list(filter(lambda x: isinstance(x, R2Conv), model.modules()))
    if not layers:
        return
    layer_ids = [layer.layer_id for layer in layers]
    layer_offset = min(layer_ids)
    eq_modules = [
        [
            (
                [layer for layer in module]
                if isinstance(module, nn.SequentialModule)
                else [module]
            )
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
            data, _ = next(iter(loader))
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


def test_nn(model, test_loader, loss_fn, dataset):
    use_amp = USE_AMP
    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
            enabled=use_amp,
        ):
            model.eval()

    loss = 0
    acc = 0
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    with torch.no_grad():
        for i, (x, targets) in enumerate(test_loader):
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                x = x.to(DEVICE)
                targets = targets.to(DEVICE).view(-1)

                preds = model(x)

                acc += calc_accuracy(preds, targets)

                loss += loss_fn(preds, targets).item()

                targets = targets.to(torch.float32)
                preds = preds.softmax(dim=-1)

                y_true = torch.cat((y_true.cpu(), targets.cpu()), 0)
                y_score = torch.cat((y_score.cpu(), preds.cpu()), 0)

    y_true = y_true.numpy()
    y_score = y_score.detach().numpy()
    data_flag = config["dataset"]
    evaluator = Evaluator(data_flag, dataset)
    metrics = evaluator.evaluate(y_score)
    print(dataset, metrics)
    model.train()

    return loss / len(test_loader), acc / len(test_loader)


def log_confusion_matrix(model, loader, config):
    label_names = [config.labels[key] for key in config.labels]

    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)
    classes = np.sort(list(np.unique(y_true)))
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1), index=label_names, columns=label_names
    )
    plt.figure(figsize=(20, 20))
    sn.heatmap(df_cm, annot=True)
    plt.yticks(rotation=0)
    wandb.log({"confusion matrix": wandb.Image(plt)})

    plt.close()


def create_config(args, iteration):
    config = dict(
        iteration=iteration,
        lr=args.lr,
        epochs=args.epochs,
        optimizer="Adam",
        mnist_type=args.mnist_type,
        batch_size=args.batch_size,
        restrict=args.restrict,
        normalization=not args.no_norm,
        group=args.group,
        learn_eq=args.learn_eq,
        L_in=args.L,
        L_out=args.L_out,
        kl_div=args.kl_div,
        kl_uniform=args.kl_uniform,
        alignment_loss=args.alignment_loss,
        mode_wandb=args.mode_wandb,
        one_eq=args.one_eq,
        channels=args.channels,
        nr_workers=args.nr_workers,
        dataset=args.dataset,
        resnet=args.resnet,
        early_stop=args.early_stop,
        ratio=args.ratio,
        activation=args.activation,
        invariant=args.invariant,
        rpp=args.RPP,
        conv_wd=args.conv_wd,
        basic_wd=args.basic_wd,
    )
    return config


def get_dataset_name(rotation, reflection):
    if rotation == 1:
        out = "O(2)" if reflection else "SO(2)"
    else:
        n = "4" if rotation else "1"
        out = f"D{n}" if reflection else f"C{n}"
    return out


def model_pipeline(config, mode_wandb):
    with wandb.init(
        project=args.project,
        config=config,
        mode=mode_wandb,
        tags=[str(DEVICE)],
        reinit=True,
    ):
        config = wandb.config
        set_seed(config.iteration)
        train_loader, test_loader, val_loader, counts = make_data_loaders(config)
        model = create_network(config)

        wandb.config["network_name"] = model.network_name
        wandb.config["nr_params"] = number_of_params(model)
        print(f'Nr of model params: {wandb.config["nr_params"]}')
        config = wandb.config
        wandb.run.name = f"{model.network_name} on {config.dataset}"
        wandb.config["run_name"] = wandb.run.name

        train_nn(model, train_loader, test_loader, val_loader, config, counts=counts)


def save_and_log_model(config, model, optimizer):
    try:
        os.makedirs("checkpoints/med_mnist", exist_ok=True)
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
        f"checkpoints/med_mnist/{config['dataset']}_{config['run_name']} iteration {config['iteration']}_{wandb.run.id}.pth",
    )
    try:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(
            f"checkpoints/med_mnist/{config['dataset']}_{config['run_name']} iteration {config['iteration']}_{wandb.run.id}.pth"
        )
        wandb.run.log_artifact(artifact)
    except:
        pass


if __name__ == "__main__":
    DEVICE = "cuda"
    USE_AMP = True
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--mode_wandb",
        type=str,
        help="mode of wandb: online, offline, disabled",
        default="online",
    )

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

    parser.add_argument("--epochs", type=int, default=25, help="nr of epochs to run")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--batch_size", type=int, default=32)

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
        "--learn_eq", action="store_true", help="learn the degree of equivariance"
    )

    parser.add_argument(
        "--no_norm", action="store_false", help="does not normalise basis if set"
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
        default=1,
    )

    parser.add_argument(
        "-kl_U",
        "--kl_uniform",
        type=float,
        default=1,
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

    parser.add_argument("--early_stop", action="store_true")

    parser.add_argument(
        "-ratio",
        "--ratio",
        type=float,
        default=1,
    )

    parser.add_argument("-p", "--project", type=str, default="med_3d_final")

    parser.add_argument(
        "--activation", default="gated", choices=["QFourier", "Fourier"]
    )

    parser.add_argument("--invariant", action="store_false")

    parser.add_argument("-rpp", "--RPP", action="store_true")

    parser.add_argument(
        "--basic_wd",
        type=float,
        default=1e-3,
        help="basic weight decay",
    )
    parser.add_argument(
        "--conv_wd",
        type=float,
        default=1e-5,
        help="equiv weight decay",
    )

    args = parser.parse_args()

    for iteration in args.iterations:
        config = create_config(args, iteration)
        model_pipeline(config, args.mode_wandb)
