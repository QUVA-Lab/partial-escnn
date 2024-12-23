import torch
import sys
import argparse
import wandb
import time

sys.path.append("..")
from torch.utils.data import DataLoader
from data.vector_dataset import VectorDataset
from networks import MLP, O2SteerableMLP

from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
from escnn2.linear import Linear
from util import get_kl_loss

import matplotlib.pyplot as plt

import numpy as np
import random

plt.switch_backend("agg")

from util import set_seed


USE_AMP = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.bfloat16 if DEVICE == "cpu" else torch.float16


def create_config(args, iteration):
    config = dict(
        iteration=iteration,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer="Adam",
        angle=args.angle,
        norm=args.norm,
        equiv=args.equiv,
        learn_eq=args.learn_eq,
        normalization=not args.no_norm,
        prelim=args.prelim,
        kl_div=args.kl_div,
        alignment_loss=args.alignment_loss,
        gated=args.gated,
        one_eq=args.one_eq,
        channel_splits=args.channel_splits,
        project=args.project,
        first=args.first,
    )
    return config


def create_network(config):
    if not config.equiv:
        model = MLP(2, 2)
    else:
        model = O2SteerableMLP(
            2,
            learn_eq=config.learn_eq,
            normalise_basis=config.normalization,
            prelim=config.prelim,
            gated=config.gated,
            one_eq=config.one_eq,
            split=config.channel_splits,
            iteration=config.iteration,
            only_first=config.first,
        )
    return model


def get_shift_loss(irrepsmaps):
    loss = 0
    for irrepmap in irrepsmaps:
        if hasattr(irrepmap, "shift_loss"):
            loss += irrepmap.shift_loss.flatten()[0]

    return loss


def get_irrepsmaps(model):
    irrepmaps = list(
        filter(lambda x: isinstance(x, IrrepsMapFourierBLact), model.modules())
    )
    return irrepmaps


def create_datasets(config):
    dataset_train = VectorDataset(noise=True)
    dataset_test = VectorDataset(nr_vecs=2000, noise=True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.iteration)

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=8
    )

    return train_loader, test_loader


# def get_kl_loss(irrepmaps):
#     irrepmaps = {irrepmap.layer_id: irrepmap for irrepmap in irrepmaps}
#     kl_loss_sum = 0, 0
#     for layer_id, irrepmap in irrepmaps.items():
#         if (layer_id - 1) not in irrepmaps:
#             kl_loss = irrepmap.kl_divergence(None)
#             uniform_cnt += 1
#         else:
#             kl_loss = irrepmap.kl_divergence(irrepmaps[layer_id - 1])
#             other_cnt += 1
#             # kl_loss = irrepmap.kl_divergence(None)
#         kl_loss_sum += kl_loss.squeeze() / len(irrepmaps)
#     return kl_loss_sum


def log_learned_equivariance(irrepmaps, epoch, config):
    layer_ids = set()
    layer_ids = set([irrepmap.layer_id for irrepmap in irrepmaps])
    if list(layer_ids) == []:
        return

    layer_offset = min(layer_ids)
    n = 100
    for irrepmap in irrepmaps:
        layer_id = irrepmap.layer_id
        prob_fn, g_elements, _ = irrepmap.get_distribution(n=n)
        prob_fn, g_elements = prob_fn[0], g_elements[0]
        data = [
            [g_element, prob, layer_id, config.run_name, config.iteration]
            for (g_element, prob) in zip(g_elements, prob_fn)
        ]

        table = wandb.Table(
            data=data,
            columns=[
                "transformation element",
                "equivariance degree",
                "layer_id",
                "run_name",
                "run_name_iter",
            ],
        )
        wandb.log(
            {
                f"Learned Degree of equivariance layer {layer_id - layer_offset}  iteration {config.iteration}": wandb.plot.line(
                    table,
                    "transformation element",
                    "equivariance degree",
                    title=f"Degree of equivariance layer {layer_id - layer_offset}",
                )
            },
            step=epoch,
        )
        plt.title(f"Degree of equivariance layer {layer_id - layer_offset}")

        plt.plot(g_elements, prob_fn)
        plt.ylim([0, 1.1 * max(prob_fn)])
        plt.ylabel("p(g)")
        plt.xlabel("groupelement g")
        wandb.log(
            {
                f"gradual Degree of equivariance layer {layer_id - layer_offset}  iteration {config.iteration}": wandb.Image(
                    plt
                )
            },
            step=epoch,
        )
        plt.close()


def log_equivariance_error(model, epoch, config):
    model.eval()
    layers = list(filter(lambda x: isinstance(x, Linear), model.modules()))
    if not layers:
        return
    layer_ids = [layer.layer_id for layer in layers]
    layer_offset = min(layer_ids)
    group = model.G
    n = 100
    testing_elements = [elem for elem in group.testing_elements(n=n)]
    factor = len(testing_elements) // n
    g_elements = np.concatenate(
        [
            np.linspace(i * (2 * np.pi), 2 * np.pi * (i + 1), n, endpoint=False)
            for i in range(factor)
        ]
    )
    with torch.no_grad():
        for layer in layers:
            layer_id = layer.layer_id
            in_type = layer.in_type
            data = in_type(torch.randn(256, in_type.size)).to(DEVICE)
            preds = layer(data)
            preds_norm = torch.linalg.norm(preds.tensor, axis=1)
            errors = []
            for t in testing_elements:
                augmentend_data = data.transform(t)
                aug_preds = layer(augmentend_data).tensor
                error = (
                    torch.linalg.norm(aug_preds - preds.transform(t).tensor, axis=1)
                    / preds_norm
                ).mean()

                errors.append(error.item())

            data = [
                [g_element, error, layer_id, config.run_name, config.iteration]
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
                    f"Equivariance error layer {layer_id - layer_offset}  iteration {config.iteration}": wandb.plot.line(
                        table,
                        "transformation element",
                        "error",
                        title=f"Equivariance error layer {layer_id - layer_offset}",
                    )
                },
                step=epoch,
            )

            plt.title(f"Equivariance error {layer_id - layer_offset}")

            plt.plot(g_elements, errors)
            try:
                plt.ylim([0, 1.1 * max(errors) + 1e-4])
            except ValueError:
                pass
            plt.ylabel("Equivariance Error")
            plt.xlabel("groupelement g")
            wandb.log(
                {
                    f"gradual Equivariance error {layer_id - layer_offset}": wandb.Image(
                        plt
                    )
                },
                step=epoch,
            )
            plt.close()
    model.train()


def train_nn(model, train_loader, test_loader, config):
    model = model.to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.watch(model, loss_fn, log="all", log_freq=1)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    irrepmaps = get_irrepsmaps(model)

    log_learned_equivariance(irrepmaps, 0, config)
    log_equivariance_error(model, 0, config)

    for epoch in range(config.epochs):
        (
            running_norm_loss,
            running_angle_loss,
            running_kl,
            running_kl_uni,
            running_shift,
        ) = (0, 0, 0, 0, 0)
        for _, (x, targets) in enumerate(train_loader):
            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=USE_AMP):
                x = x.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(x)

                loss = 0
                if config.norm:
                    # print("x", x[0])
                    # print("norm/angle", targets[0])
                    loss_norm = loss_fn(preds[:, 0], targets[:, 0])
                    running_norm_loss += loss_norm.item() / len(train_loader)
                    loss += loss_norm

                if config.angle:
                    loss_angle = loss_fn(
                        torch.cos(preds[:, 1]), torch.cos(targets[:, 1])
                    )

                    running_angle_loss += loss_angle.item() / len(train_loader)
                    loss += loss_angle
                shift_loss = get_shift_loss(irrepmaps)
                kl_loss, kl_uniform = get_kl_loss(irrepmaps)
                running_shift += shift_loss / len(train_loader)
                running_kl += kl_loss / len(train_loader)
                running_kl_uni += kl_uniform / len(train_loader)
                loss += (
                    kl_loss * config.kl_div
                    + kl_uniform * config.kl_div
                    + shift_loss * config.alignment_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        test_norm, test_angle, identity_distances = test_nn(
            model, test_loader, loss_fn, config, epoch
        )
        wandb.log(
            {
                "Train norm loss": running_norm_loss,
                "Train angle loss": running_angle_loss,
                "Train total loss": running_norm_loss + running_angle_loss,
                "KL-divergence": running_kl,
                "KL-uni": running_kl_uni,
                "Alignment Loss": running_shift,
                "Test norm loss": test_norm,
                "Test angle loss": test_angle,
                "Test total loss": test_norm + test_angle,
            }
            | {
                f"Layer {i+1} non-equivariance": distance
                for i, distance in enumerate(identity_distances)
            },
            step=epoch + 1,
        )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch: {epoch} \t train_losses:  norm: {running_norm_loss:.3f} / angle: {running_angle_loss:.3f} \t test_loss: norm: {test_norm:.3f} / angle: {test_angle:.3f}"
            )
        if (epoch + 1) % 5 == 0:
            log_learned_equivariance(irrepmaps, epoch + 1, config)
            log_equivariance_error(model, epoch + 1, config)


def test_nn(model, test_loader, loss_fn, config, epoch):
    model.eval()
    loss_norm = 0
    loss_angle = 0
    with torch.no_grad():
        for _, (x, targets) in enumerate(test_loader):
            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=USE_AMP):
                x = x.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(x)

                if args.norm:
                    loss_norm += loss_fn(preds[:, 0], targets[:, 0]).item() / len(
                        test_loader
                    )
                if args.angle:
                    loss_angle += loss_fn(
                        torch.cos(preds[:, 1]), torch.cos(targets[:, 1])
                    ).item() / len(test_loader)
        identity_distances = []
        if config.learn_eq:
            model_params = filter(
                lambda p: p.shape == torch.Size((2, 2)), model.parameters()
            )

            for parameter in model_params:
                param = parameter.detach().cpu()
                diag_mean = torch.diagonal(param, 0).mean()
                param -= diag_mean * torch.eye(n=param.shape[0], m=param.shape[1])
                identity_distances.append(
                    torch.linalg.norm(param.flatten()).item() ** 2
                )

    model.train()
    return loss_norm, loss_angle, identity_distances


def create_run_name(config, model):
    out = model.network_name
    out += f" On "
    if not config.angle:
        out += "Norm" if config.norm else ""
    else:
        out += "Angle and Norm" if config.norm else "Angle"
    return out


def number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_pipeline(config, mode_wandb):
    with wandb.init(
        project=config["project"],
        config=config,
        mode=mode_wandb,
        tags=[str(DEVICE)],
        reinit=True,
    ):
        config = wandb.config

        set_seed(config.iteration)

        model = create_network(config)
        print(f"Number of params: {number_of_params(model)}")
        wandb.run.name = create_run_name(config, model)
        config.run_name = wandb.run.name
        train_loader, test_loader = create_datasets(config)
        start = time.time()
        train_nn(model, train_loader, test_loader, config)
        print(time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode_wandb",
        type=str,
        help="mode of wandb: online, offline, disabled",
        default="online",
    )

    # parser.add_argument(
    #     "-e",
    #     "--equivariance",
    #     type=bool,
    #     choices=["no", "steer", "learn", "learn_norm", "learn_norm_wigner"],
    #     default="learn_norm",
    # )

    parser.add_argument(
        "-e", "--equiv", action="store_true", help="use (partial) equivariant mapping"
    )

    parser.add_argument(
        "--learn_eq", action="store_true", help="learn the degree of equivariance"
    )

    parser.add_argument(
        "--no_norm", action="store_false", help="does not normalise basis if set"
    )

    parser.add_argument(
        "--prelim", action="store_true", help="set to use preliminary approach"
    )

    parser.add_argument("--epochs", type=int, default=100, help="nr of epochs to run")

    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")

    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("-a", "--angle", action="store_true")
    parser.add_argument("-n", "--norm", action="store_true")

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        nargs="*",
        help="Number of iterations to run",
        default=[0],
    )

    parser.add_argument(
        "-kl",
        "--kl_div",
        type=float,
        default=25,
    )

    parser.add_argument("--one_eq", action="store_true")

    parser.add_argument("-c", "--channel_splits", action="store_true")

    parser.add_argument("-align", "--alignment_loss", type=float, default=1)

    parser.add_argument("-g", "--gated", action="store_true")

    parser.add_argument("-p", "--project", type=str, required=True)

    parser.add_argument("--first", action="store_true")

    args = parser.parse_args()
    assert args.norm or args.angle, "cannot both be false"

    for iteration in args.iterations:
        config = create_config(args, iteration)
        model_pipeline(config, args.mode_wandb)
