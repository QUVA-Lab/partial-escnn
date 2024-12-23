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


import matplotlib.ticker as ticker


# def create_ticker(i):
#     # Create a FuncFormatter.
#     return ticker.FuncFormatter(multiple_formatter)


network = "preliminary_norm_gated"
dir = f"{network}"
os.makedirs(dir, exist_ok=True)
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(24, 6), layout="constrained")

# ticks = ticks = map(create_ticker, range(3))

for l, (host) in enumerate(axes):
    # host = ax
    file = pandas.read_csv(f"{network}/{network}_layer_{l}_error.csv")

    y = np.array(file["error"])

    shift = np.argmax(y[: len(y) // 2])
    elements = len(y) // 2
    # y[:elements] = np.roll(y[:elements], elements - shift)
    # y[elements:] = np.roll(y[elements:], elements - shift)
    x_plot = np.array(file["transformation element"])
    p1 = host.plot(x_plot, y, label="error", color="orange")
    host.set_ylim(0, 2.6)
    host.set_title(f"Layer {l+1}")
    host.set_xlim(0, 4 * np.pi)
    host.plot(
        [2 * np.pi, 2 * np.pi], [0, 100], color="black", alpha=0.5, linestyle="dashed"
    )

    host.set_yticks(np.linspace(0, 2.5, 6))

    # ax = fig.gca()
    host.set_xlabel(r"$h \in H$")
    host.set_ylabel(f"equivariance error")
    host.axhline(0, color="black", lw=2)
    host.axvline(0, color="black", lw=2)
    host.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
    host.xaxis.set_minor_locator(ticker.MultipleLocator(np.pi / 12))
    host.xaxis.set_major_formatter(ticker.FuncFormatter(multiple_formatter()))

    for i, label in enumerate(host.get_xticklabels()):
        if i > len(host.get_xticklabels()) // 2:
            label.set_color("red")
        else:
            label.set_color("blue")

plt.savefig(f"{dir}/{network}.pdf", bbox_inches="tight")
# plt.close()
