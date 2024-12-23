import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
import os

plt.rcParams.update({"font.size": 18})

plt.figure(figsize=(10, 6))


dir = f"kl_epochs"

file = pandas.read_csv(f"{dir}/kl.csv")

name = file.columns[1]

data = file[name]

file2 = pandas.read_csv(f"{dir}/no_kl.csv")

name2 = file2.columns[1]

data2 = file2[name2]

plt.plot(data, label="With KL regularisation")

plt.plot(data2, label="Without KL regularisation")

plt.ylabel("KL-divergence")

plt.xlabel("epochs")
ax = plt.gca()
ax.axhline(0, color="black", lw=2)
ax.axvline(0, color="black", lw=2)

plt.xlim(0, 100)
plt.ylim(0)

plt.legend()

plt.savefig(f"{dir}/kl_plot.pdf", bbox_inches="tight")
