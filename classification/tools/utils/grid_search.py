alpha_range = [1, 0.7, 0.5]
beta_range = [0.5, 0.7, 0.9]
import os
# for trial in range(3):
for a in alpha_range:
    for b in beta_range:
        cmd = "python3 tools/train/train_ELR_loss.py -i config/mitotic_simple_extend_ELR.py -o classification_stage/ELR/{}_{}_2 -a {} -b {}".format(a, b, a, b)
        print(cmd)
        os.system(cmd)

# params = [(0.75, 0.2), (0.25, 1), (0.5, 0.5)]
# import os
# for trial in range(3):
#     for para in params:
#         a, b = para
#         cmd = "python3 tools/train_torch.py -i config/mitotic_simple.py -o torch/baseline/focal_loss/alpha={},beta={},trial={} -a {} -b {}".format(a, b, trial, a, b)
#         print(cmd)
#         os.system(cmd)
