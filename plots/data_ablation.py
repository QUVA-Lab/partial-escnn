import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18})


x_orig = np.array([0.0625, 0.125, 0.250, 0.5, 0.75, 1][::-1])

nr_imgs = {"OrganMNIST3D": 972, "NoduleMNIST3D": 1158}

dataset = "OrganMNIST3D"
x = x_orig * nr_imgs[dataset]
if dataset == "OrganMNIST3D":
    learn_eq_vals = np.array([0.906, 0.853, 0.788, 0.755, 0.651, 0.511])

    learn_eq_std = np.array([0.002, 0.025, 0.057, 0.029, 0.009, 0.083])

    eq_vals = np.array([0.604, 0.578, 0.521, 0.459, 0.395, 0.323])

    eq_std = np.array([0.013, 0.014, 0.024, 0.013, 0.011, 0.018])

    cnn_vals = np.array([0.911, 0.897, 0.858, 0.821, 0.759, 0.659])

    cnn_std = np.array([0.018, 0.024, 0.050, 0.011, 0.025, 0.024])

    rpp_vals = np.array([0.939, 0.919, 0.889, 0.857, 0.772, 0.678])

    rpp_std = np.array([0.004, 0.017, 0.022, 0.006, 0.009, 0.010])

elif dataset == "NoduleMNIST3D":
    learn_eq_vals = np.array([0.855, 0.851, 0.839, 0.843, 0.831])

    learn_eq_std = np.array([0.014, 0.012, 0.008, 0.011, 0.010])

    eq_vals = np.array([0.854, 0.834, 0.828, 0.827, 0.752])

    eq_std = np.array([0.014, 0.003, 0.003, 0.034, 0.098])

    cnn_vals = np.array([0.840, 0.832, 0.819, 0.810, 0.774])

    cnn_std = np.array([0.030, 0.006, 0.021, 0.007, 0.005])

    rpp_vals = np.array([0.849, 0.828, 0.827, 0.818, 0.810])

    rpp_std = np.array([0.019, 0.014, 0.010, 0.010, 0.006])
plt.plot(x, learn_eq_vals, color="blue", label=r"$SO(3)$ P-SCNN (Ours)")

plt.fill_between(
    x,
    learn_eq_vals + learn_eq_std / 2,
    learn_eq_vals - learn_eq_std / 2,
    facecolor="blue",
    alpha=0.15,
)
plt.plot(x, eq_vals, color="orange", label=r"$SO(3)$ SCNN")

plt.fill_between(
    x, eq_vals + eq_std / 2, eq_vals - eq_std / 2, facecolor="orange", alpha=0.15
)

plt.plot(x, cnn_vals, color="red", label=r"CNN")

plt.fill_between(
    x, cnn_vals + cnn_std / 2, cnn_vals - cnn_std / 2, facecolor="red", alpha=0.15
)

plt.plot(x, rpp_vals, color="green", label=r"$SO(3)$ RPP")

plt.fill_between(
    x,
    rpp_vals + rpp_std / 2,
    rpp_vals - rpp_std / 2,
    facecolor="green",
    alpha=0.15,
)

# plt.xticks()
plt.xlim(0)
# plt.ylim(0)
# plt.axhline(0, color="black", lw=2)
# plt.axvline(0, color="black", lw=2)
if dataset == "NoduleMNIST3D":
    plt.ylabel("Accuracy")
plt.xlabel("# training samples")
# plt.legend(loc="lower right")
plt.savefig(f"ablation_{dataset}.pdf", bbox_inches="tight")
