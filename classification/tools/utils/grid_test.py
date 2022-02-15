# alpha_range = [0.1, 0.3, 0.5, 0.7]
# beta_range = [0.5, 0.7, 0.9]
# import os
# for trial in range(5):
#     for a in alpha_range:
#         for b in beta_range:
#             cmd = "python3 tools/test_torch.py -i config/mitotic.py -m torch/ELR/lambda={},beta={},trial={}".format(a, b, trial)
#             print(cmd)
#             os.system(cmd)

# params = [(0.1, 0.7), (3, 0.9)]
# import os
# for trial in range(3):
#     for para in params:
#         a, b = para
#         cmd = "python3 tools/test_torch.py -i config/mitotic.py -m torch/ELR_effnet_4cls_certain/lambda={},beta={},trial={}".format(a, b, trial)
#         print(cmd)
#         os.system(cmd)

params = [(0.75, 0.2), (0.25, 1), (0.5, 0.5)]
import os
for trial in range(3):
    for para in params:
        a, b = para
        cmd = "python3 tools/test_torch.py -i config/mitotic_simple.py -m torch/baseline/focal_loss/alpha={},beta={},trial={}".format(a, b, trial)
        print(cmd)
        os.system(cmd)
