import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
import os

plt.rcParams.update({"font.size": 28})


# np.random.seed(1)

# Ls = 16

# x = np.linspace(0, 2 * np.pi, 1000, False)

# coeff = np.random.uniform(0.1, 1, Ls * 2 + 1)
# coeff2 = np.random.uniform(0.1, 1, Ls * 2 + 1)

# y1 = np.zeros_like(x)
# y2 = np.zeros_like(x)
# y1 += coeff[0]
# y2 += coeff2[0]


def normalise(prob):
    if np.min(prob) < 0:
        prob -= np.min(prob)
    prob /= np.mean(prob)
    return prob


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        refl = False
        if x > 2 * np.pi:
            refl = True
            x -= 2 * np.pi
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (
                    num,
                    latex,
                    den,
                )

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


network = "angle_gated"
dir = f"{network}"
os.makedirs(dir, exist_ok=True)
nr_things = 3
fig, host_orig = plt.subplots(
    ncols=nr_things, nrows=1, figsize=(8 * nr_things, 6), layout="constrained"
)
for l in range(nr_things):
    if nr_things > 1:
        host = host_orig.flatten()[l]
    else:
        host = host_orig

    file = pandas.read_csv(f"{network}/{network}_layer_{l}.csv")

    y = np.array(file["equivariance degree"])

    shift = np.argmax(y[: len(y) // 2])
    elements = len(y) // 2
    y[:elements] = np.roll(y[:elements], elements - shift)
    y[elements:] = np.roll(y[elements:], elements - shift)
    x_plot = np.array(file["transformation element"])
    p1 = host.plot(
        x_plot,
        y,
        label="likelihood",
    )
    host.set_ylim(0, 2.6)
    host.set_title(f"Layer {l+1}")
    plt.xlim(0, 4 * np.pi)
    plt.plot(
        [2 * np.pi, 2 * np.pi], [0, 100], color="black", alpha=0.5, linestyle="dashed"
    )
    # if not "oneEq" in network:
    ax2 = host.twinx()
    file = pandas.read_csv(f"{network}/{network}_layer_{l}_error.csv")

    # file_2 = pandas.read_csv(f"{network}/{network}_layer_{l}_error_Full.csv")

    y_2 = np.array(file["error"])  # - np.array(file_2["error"])
    x_plot = np.array(file["transformation element"])
    p2 = ax2.plot(
        x_plot,
        y_2,
        color="orange",
        label="error difference",
    )
    ax2.set_ylim(0, np.max(y_2) / (np.max(y) / 2.6))
    # plt.ylim(0, 2.6)
    plt.xlim(0, 4 * np.pi)

    if not l:
        host.set_ylabel(r"$\lambda(h)$")
        # print(np.linspace(0, 2.5, 6))
        host.set_yticks(np.linspace(0, 2.5, 6))
    else:
        host.tick_params(left=False, labelleft=False)
    host.plot(
        [2 * np.pi, 2 * np.pi],
        [0, 100],
        color="black",
        alpha=0.5,
        linestyle="dashed",
    )

    host.legend(handles=p1 + p2, loc="best")
    ax2.set_ylabel(f"error difference")
    # ax2.legend()

    ax = plt.gca()
    host.set_xlabel(r"$h \in H$")
    host.set_ylabel(r"$\lambda(h)$")

    ax.axhline(0, color="black", lw=2)
    ax.axvline(0, color="black", lw=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for i, label in enumerate(host.get_xticklabels()):
        if i > len(host.get_xticklabels()) // 2:
            label.set_color("red")
        else:
            label.set_color("blue")

plt.savefig(f"{dir}/{network}.pdf", bbox_inches="tight")
# plt.close()
