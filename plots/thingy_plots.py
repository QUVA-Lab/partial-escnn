import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 24})

elements = ["a", "b", "c", "d"]

vals = [[1, 1, 1, 1], [2, 2, 0.01, 0.01], [2.1, 1.9, 0.01, 0.01]]


f, axes = plt.subplots(ncols=3, nrows=1, figsize=(24, 6), layout="constrained")
# f.ylim(0, 2.5)
for i, ax in enumerate(axes):
    ax.plot(
        elements,
        vals[i],
    )

    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, "d")
    ax.set_xlabel(r"$h\in H$")

axes[0].set_ylabel(r"$\lambda(h)$")

plt.savefig("synthetic_likelihoods.pdf", bbox_inches="tight")
