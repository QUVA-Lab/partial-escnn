import os

hparams_file = "hparams_vector.txt"

with open(hparams_file) as f:
    for i, line in enumerate(f.readlines()):
        os.system(f"python vector_mlp.py {line[:-1]}")
