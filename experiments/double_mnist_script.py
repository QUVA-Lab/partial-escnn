import os


# f = open("hparams_double_Fourier.txt")

# for i, line in enumerate(f.readlines()[19:]):
#     print("##################")
#     print(f"LINE {i+1}!!!!!")
#     print("##################")
#     os.system(f"python double_mnist.py {line[:-1]}")


f = open("hparams_med_ablation.txt")

for i, line in enumerate(f.readlines()[59:61]):
    print("##################")
    print(f"LINE {i+1}!!!!!")
    print("##################")
    # print(line[:-1])
    os.system(f"python medical_mnist2d.py {line[:-1]}")
