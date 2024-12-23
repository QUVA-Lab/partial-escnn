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
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from data import MNIST_Double
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
from escnn2.r2convolution import R2Conv
from networks import CNN, SteerableCNN, SteerableRPP, RPPBlock
import random

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
    transform_list = [transforms.ConvertImageDtype(torch.float)]
    if config.data_rotation == 0:
        transform_list.append(Rotate90Transform(angles=[0]))
    elif config.data_rotation == 1:
        transform_list.append(
            transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
        )
    elif config.data_rotation == 2:
        transform_list.append(Rotate90Transform())
    elif config.data_rotation == 3:
        transform_list.append(Rotate90Transform(angles=[0, 120, 240]))
    if config.data_reflection:
        transform_list.append(transforms.RandomHorizontalFlip())
    aug_transform = transforms.Compose(transform_list)

    mnist_test_aug = MNIST_Double(
        train=False, digit_transform=aug_transform, images_per_class=50
    )

    mnist_test = MNIST_Double(train=False, images_per_class=50)

    mnist_validate = MNIST_Double(
        train=False, digit_transform=aug_transform, images_per_class=20
    )

    mnist_train = MNIST_Double(
        digit_transform=aug_transform, images_per_class=int(100 * config.ratio)
    )

    aug_loader = torch.utils.data.DataLoader(
        mnist_test_aug, batch_size=config.batch_size, num_workers=config.nr_workers
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=config.batch_size, num_workers=config.nr_workers
    )

    validate_loader = torch.utils.data.DataLoader(
        mnist_validate, batch_size=config.batch_size, num_workers=config.nr_workers
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.iteration)

    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.nr_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    aug_test_loaders = None
    if config.data_rotation in [2, 3] and config.group in ["SO2", "O2", "CNN"]:
        aug_test_loaders = {}
        for rotation in [1]:
            for reflection in [0, 1]:
                transform_list = [transforms.ConvertImageDtype(torch.float)]
                if rotation == 0:
                    transform_list.append(Rotate90Transform(angles=[0]))
                elif rotation == 1:
                    transform_list.append(
                        transforms.RandomRotation(
                            180, interpolation=InterpolationMode.BILINEAR
                        )
                    )
                elif rotation == 2:
                    transform_list.append(Rotate90Transform())
                elif rotation == 3:
                    transform_list.append(Rotate90Transform(angles=[0, 120, 240]))
                if reflection:
                    transform_list.append(transforms.RandomHorizontalFlip())

                aug_transform = transforms.Compose(transform_list)
                dataset = MNIST_Double(
                    train=False, digit_transform=aug_transform, images_per_class=50
                )
                name = get_dataset_name(rotation, reflection)
                aug_test_loaders[name] = torch.utils.data.DataLoader(
                    dataset, batch_size=config.batch_size, num_workers=16
                )

    return train_loader, test_loader, aug_loader, aug_test_loaders, validate_loader


def create_network(config):
    if config.group == "CNN":
        return CNN(mnist_type=config.mnist_type, n_classes=100).to(DEVICE)
    elif config.rpp:
        return SteerableRPP.from_group(
            config.group,
            mnist_type=config.mnist_type,
            restrict=config.restrict,
            n_classes=100,
            one_eq=config.one_eq,
            split=config.channel_splits,
            iteration=config.iteration,
            activation=(
                nn.GatedNonLinearityUniform if not config.fourier else nn.FourierELU
            ),
            invariant=config.invariant,
        )
    else:
        return SteerableCNN.from_group(
            config.group,
            mnist_type=config.mnist_type,
            restrict=config.restrict,
            n_classes=100,
            learn_eq=config.learn_eq,
            normalise_basis=config.normalization,
            one_eq=config.one_eq,
            split=config.channel_splits,
            iteration=config.iteration,
            L_in=config.L_in,
            L_out=config.L_out,
            activation=(
                nn.GatedNonLinearityUniform if not config.fourier else nn.FourierELU
            ),
            invariant=config.invariant,
        ).to(DEVICE)


def train_nn(
    model,
    train_loader,
    test_loader,
    aug_loader,
    val_loader,
    config,
    augment_test_loaders=None,
):
    use_amp = USE_AMP
    epochs = config.epochs
    model = model.to(DEVICE)
    model.train()
    # inputs = torch.randn(1024, 1, 28, 28).to(DEVICE)
    # model(inputs)
    # inputs = torch.randn(1024, 1, 28, 28).to(DEVICE)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    # ) as prof:
    #     with record_function("model_inference"):
    #         start = time.time()
    #         model(inputs)
    #         end = time.time()
    #         print(end - start)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # exit()

    loss_fn = torch.nn.CrossEntropyLoss()

    if config.rpp:
        regularizer = RPPConv_L2
    else:
        regularizer = None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.watch(model, loss_fn, log="all", log_freq=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    irrepmaps = get_irrepsmaps(model)

    log_learned_equivariance(irrepmaps, 0, config)

    max_validate_acc, max_test_acc = 0, 0

    for epoch in range(epochs):
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
                x = x.to(DEVICE)
                targets = targets.to(DEVICE)
                start = time.time()

                preds = model(x)

                end = time.time()

                # print(end - start)
                running_acc += calc_accuracy(preds, targets)

                loss = loss_fn(preds, targets)
                running_ep_loss += loss.item()
                kl_loss, kl_uniform = get_kl_loss(irrepmaps)
                running_kl_loss += kl_loss
                running_kl_uni += kl_uniform
                shift_loss = get_shift_loss(irrepmaps)
                running_shift_loss += shift_loss

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
        print("########################################################")
        print(
            f"Running epoch loss: {running_ep_loss/(i+1):.4f}, running acc: {running_acc/(i+1):.4f}"
        )
        print(f"Epoch time: {time.time() - start_time :.4f}")

        test_loss, test_acc = test_nn(model, test_loader, loss_fn)
        val_loss_loss, val_aug_acc = test_nn(model, val_loader, loss_fn)
        test_aug_loss, test_aug_acc = test_nn(model, aug_loader, loss_fn)

        if val_aug_acc > max_validate_acc:
            max_test_acc = test_acc
            max_aug_test_acc = test_aug_acc
            max_validate_acc = val_aug_acc

        print(
            f"Epoch {epoch}: test loss {test_loss:.4f} test acc:{test_acc:.4f} aug loss: {test_aug_loss:.4f} aug test acc: {test_aug_acc:.4f}"
        )
        print("\n########################################################")
        wandb.log(
            {
                "Train Loss": running_ep_loss / (i + 1),
                "Alignment Loss": running_shift_loss / (i + 1),
                "KL-Divergence": running_kl_loss / (i + 1),
                "Train Accuracy": running_acc / (i + 1),
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "AugTest Loss": test_aug_loss,
                "AugTest Accuracy": test_aug_acc,
                "Val Loss": val_loss_loss,
                "Val Accuracy": val_aug_acc,
                "max AugTest Accuracy": max_aug_test_acc,
                "max Test Accuracy": max_test_acc,
                "Running KL-Divergence": running_kl_loss / (i + 1),
                "KL-Divergence": kl_loss,
                "Running KL-Divergence_uniform": running_kl_uni / (i + 1),
                "KL-Divergence-uniform": kl_uniform,
            },
            step=epoch + 1,
        )

        if augment_test_loaders is not None:
            for symmetry, loader in augment_test_loaders.items():
                test_loss, test_acc = test_nn(model, loader, loss_fn)
                print(symmetry, test_acc)
                wandb.log(
                    {
                        f"{symmetry} Test Accuracy": test_acc,
                    },
                    step=epoch + 1,
                )

        log_learned_equivariance(irrepmaps, epoch + 1, config)

        log_confusion_matrix(model, aug_loader)
    if config.iteration == 0:
        log_equivariance_error(model, config, test_loader)
    save_and_log_model(dict(config), model, optimizer)


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

                pred_norms = torch.linalg.norm(preds.tensor, axis=1, keepdim=True)
                x_shape = list(x.shape)[2]

                if isinstance(layer, R2Conv) or isinstance(layer, RPPBlock):
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

                            pred_norms_channel = pred_norms[
                                :, start_channel : start_channel + channel.size, :, :
                            ]

                            channel_errors.append(
                                (
                                    torch.linalg.norm(
                                        aug_preds_channel - preds_transform_channel,
                                        axis=1,
                                        keepdim=True,
                                    )
                                    / pred_norms_channel
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


def test_nn(model, test_loader, loss_fn):
    use_amp = USE_AMP
    model.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for i, (x, targets) in enumerate(test_loader):
            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if DEVICE == "cpu" else torch.float16,
                enabled=use_amp,
            ):
                x = x.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(x)
                acc += calc_accuracy(preds, targets)

                loss += loss_fn(preds, targets).item()

    model.train()

    return loss / len(test_loader), acc / len(test_loader)


def log_confusion_matrix(model, loader):
    model = model.to(DEVICE)
    model.eval()
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
        cf_matrix / np.sum(cf_matrix, axis=1), index=classes, columns=classes
    )
    plt.figure(figsize=(20, 20))
    sn.heatmap(df_cm, annot=False)
    wandb.log({"confusion matrix": wandb.Image(plt)})
    plt.close()
    model.train()


def create_config(args, iteration):
    config = dict(
        iteration=iteration,
        lr=args.lr,
        epochs=args.epochs,
        optimizer="Adam",
        mnist_type=args.mnist_type,
        batch_size=args.batch_size,
        restrict=args.restrict,
        learn_eq=args.learn_eq,
        normalization=not args.no_norm,
        group=args.group,
        data_reflection=args.data_reflection,
        data_rotation=args.data_rotation,
        L_in=args.L,
        L_out=args.L_out,
        kl_div=args.kl_div,
        kl_uniform=args.kl_uniform,
        alignment_loss=args.alignment_loss,
        activation="GatedNonLinearityUniform",
        mode_wandb=args.mode_wandb,
        one_eq=args.one_eq,
        channel_splits=args.channel_splits,
        nr_workers=args.nr_workers,
        ratio=args.ratio,
        project=args.project,
        rpp=args.RPP,
        conv_wd=args.conv_wd,
        basic_wd=args.basic_wd,
        fourier=args.Fourier,
        invariant=args.invariant,
    )
    return config


def get_dataset_name(rotation, reflection):
    if rotation == 1:
        out = "O(2)" if reflection else "SO(2)"
    else:
        if not rotation:
            n = "1"
        else:
            n = "4" if rotation == 2 else "3"
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
        model = create_network(config)
        (
            train_loader,
            test_loader,
            aug_loader,
            aug_test_loaders,
            val_loader,
        ) = make_data_loaders(config)
        wandb.config["network_name"] = model.network_name
        wandb.config["dataset_symmetries"] = get_dataset_name(
            config.data_rotation, config.data_reflection
        )
        wandb.config["nr_params"] = number_of_params(model)
        print(f'Nr of model params: {wandb.config["nr_params"]}')
        config = wandb.config
        wandb.run.name = f"{model.network_name} on {config.dataset_symmetries}"
        wandb.config["run_name"] = wandb.run.name

        train_nn(
            model,
            train_loader,
            test_loader,
            aug_loader,
            val_loader,
            config,
            aug_test_loaders,
        )


def save_and_log_model(config, model, optimizer):
    model.train()
    try:
        os.makedirs("checkpoints/mnist_double/", exist_ok=True)
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
        f"checkpoints/mnist_double/{config['run_name']} iteration {config['iteration']}.pth",
    )
    try:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(
            f"checkpoints/mnist_double/{config['run_name']} iteration {config['iteration']}.pth"
        )
        wandb.run.log_artifact(artifact)
    except:
        pass


if __name__ == "__main__":
    DEVICE = "cuda"
    USE_AMP = False
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
        "--mnist_type", type=str, choices=["single", "double"], default="double"
    )

    parser.add_argument("--epochs", type=int, default=50, help="nr of epochs to run")

    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--nr_workers", type=int, default=12)

    parser.add_argument(
        "--data_reflection",
        action="store_true",
        help="individual digits are flipped if set to true",
    )

    parser.add_argument(
        "--data_rotation", type=int, help="Degrees to rotate digits with", default=0
    )

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
        default=4,
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
        "-kl_U",
        "--kl_uniform",
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

    parser.add_argument(
        "-ratio",
        "--ratio",
        type=float,
        default=1,
    )

    parser.add_argument("-rpp", "--RPP", action="store_true")

    parser.add_argument("-p", "--project", type=str, default="double_mnist_Final")

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

    parser.add_argument("--Fourier", action="store_true")

    parser.add_argument("--invariant", action="store_false")

    args = parser.parse_args()
    for iteration in args.iterations:
        config = create_config(args, iteration)
        model_pipeline(config, args.mode_wandb)
