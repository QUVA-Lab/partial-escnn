import torch
import sys

sys.path.append("..")
from torch.utils.data import DataLoader
from data.vector_dataset import VectorDataset
from networks import MLP, O2SteerableMLP
import time

torch.autograd.set_detect_anomaly(True)

USE_AMP = False
device = "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16


def train_nn(model, train_loader, test_loader, epochs=25):
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for epoch in range(epochs):
        running_norm_loss, running_angle_loss = 0, 0
        for i, (x, targets) in enumerate(train_loader):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=USE_AMP):
                x = x.to(device)
                targets = targets.to(device)

                preds = model(x)

                loss = 0

                if NORM:
                    loss_norm = loss_fn(preds[:, 0], targets[:, 0])
                    running_norm_loss += loss_norm.item() / len(train_loader)
                    loss += loss_norm

                if ANGLE:
                    loss_angle = loss_fn(
                        torch.cos(preds[:, 1]), torch.cos(targets[:, 1])
                    )

                    running_angle_loss += loss_angle.item() / len(train_loader)
                    loss += loss_angle

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        test_loss_norm, test_loss_angle, identity_dist = test_nn(
            model, test_loader, loss_fn
        )
        if epoch % 5 == 0 or epoch == epochs - 1 or 1:
            # test_loss = test_nn(model, test_loader, loss_fn)
            print(
                f"epoch: {epoch} \t train_losses:  norm: {running_norm_loss:.3f} / angle: {running_angle_loss:.3f} \t test_loss: norm: {test_loss_norm:.3f} / angle: {test_loss_angle:.3f} \t identity distances {[float('%.4f' % dist) for dist in identity_dist ]}"
            )


def test_nn(model, test_loader, loss_fn):
    model.eval()
    loss_norm = 0
    loss_angle = 0
    with torch.no_grad():
        for i, (x, targets) in enumerate(test_loader):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=USE_AMP):
                x = x.to(device)
                targets = targets.to(device)

                preds = model(x)

                if NORM:
                    loss_norm += loss_fn(preds[:, 0], targets[:, 0]).item() / len(
                        test_loader
                    )
                if ANGLE:
                    loss_angle += loss_fn(
                        torch.cos(preds[:, 1]), torch.cos(targets[:, 1])
                    ).item() / len(test_loader)

        model_params = filter(
            lambda p: p.shape == torch.Size((2, 2)), model.parameters()
        )
        identity_distances = []
        for parameter in model_params:
            param = parameter.detach().cpu()
            diag_mean = torch.diagonal(param, 0).mean()
            param -= diag_mean * torch.eye(n=param.shape[0], m=param.shape[1])
            identity_distances.append(torch.linalg.norm(param.flatten()).item() ** 2)

    model.train()
    return loss_norm, loss_angle, identity_distances


if __name__ == "__main__":
    USE_AMP = False
    device = "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    # model = O2SteerableMLP(2)
    # model = O2SteerableMLP(2, basisexpansion="learn_eq")
    # model = MLP(2, 2)

    dataset_train = VectorDataset(noise=True)
    dataset_test = VectorDataset(nr_vecs=2000, noise=True)

    train_loader = DataLoader(
        dataset_train, batch_size=1024, shuffle=True, num_workers=8
    )

    test_loader = DataLoader(
        dataset_test, batch_size=1024, shuffle=False, num_workers=8
    )

    # Flags whether to train on vector norm, angle or both
    NORM = False
    ANGLE = True
    models = [
        (
            O2SteerableMLP(2, learn_eq=True),
            " With Learnable Equivariance",
        ),
        (
            O2SteerableMLP(2, learn_eq=True, prelim=True),
            " With Learnable Equivariance",
        ),
        (O2SteerableMLP(2), ""),
        (MLP(2, 2), ""),
    ]
    for model, setup in models:
        print(f"Running for: {model.__class__.__name__} {setup}\n")
        for parameter in model.parameters():
            if parameter.shape == torch.Size((2, 2)) and not isinstance(model, MLP):
                print(parameter)
        start = time.time()
        train_nn(model, train_loader, test_loader)
        print(time.time() - start)
        # for parameter in model.parameters():
        #     if parameter.shape == torch.Size((2, 2)) and not isinstance(model, MLP):
        #         print(
        #             (
        #                 parameter.detach().cpu()
        #                 - torch.eye(n=parameter.shape[0], m=parameter.shape[1])
        #             ).sum()
        #         )

        print("")
