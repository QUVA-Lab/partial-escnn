import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})

np.random.seed(1)

Ls = 16

x = np.linspace(0, 2 * np.pi, 1000, False)

coeff = np.random.uniform(0.1, 1, Ls * 2 + 1)
coeff2 = np.random.uniform(0.1, 1, Ls * 2 + 1)

y1 = np.zeros_like(x)
y2 = np.zeros_like(x)
y1 += coeff[0]
y2 += coeff2[0]


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


for L in range(Ls + 1):
    y1 += coeff[L + 1] * np.sin((L) * x) + coeff[(L + 1) + Ls - 1] * np.cos((L) * x)
    y2 += coeff2[L + 1] * np.sin((L) * x) + coeff2[(L + 1) + Ls - 1] * np.cos((L) * x)
    y = (
        np.concatenate([y1, y2])
        if np.max(y1) > np.max(y2)
        else np.concatenate([y2, y1])
    )
    y = normalise(y)
    x_plot = np.concatenate([x, x + 2 * np.pi])
    plt.plot(x_plot, y)
    plt.ylim(0, 2.6)
    plt.xlim(0, 4 * np.pi)
    plt.plot(
        [2 * np.pi, 2 * np.pi], [0, 100], color="black", alpha=0.5, linestyle="dashed"
    )
    plt.xlabel(r"$h \in H$")
    ax = plt.gca()
    # ax.grid(True)
    ax.axhline(0, color="black", lw=2)
    ax.axvline(0, color="black", lw=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for i, label in enumerate(ax.get_xticklabels()):
        if i > len(ax.get_xticklabels()) // 2:
            label.set_color("red")
        else:
            label.set_color("blue")

    plt.savefig(f"{L}.pdf", bbox_inches="tight")
    plt.close()
