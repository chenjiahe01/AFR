from secml_malware.models.malconv2 import MalConvGCT
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
device = "cuda:0"
model = MalConvGCT(channels=256, window_size=256, stride=64,).to(device)
x = torch.load("/home/cjh/AFR/secml_malware/models/malconvGCT_nocat.checkpoint")
model.load_state_dict(x['model_state_dict'], strict=False)

model.eval()
def detect(path) :
    with open(path, 'rb') as f:
        x = f.read(4000000)
        x = x + bytes([0]) * (4000000-len(x))
        # Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
        # So decode as uint8 (1 byte per value), and then convert
        x = np.frombuffer(x, dtype=np.uint8).astype(
            np.int16)+1  # index 0 will be special padding index
        x = [x]
        x = np.array(x)
        x = torch.tensor(x)
        x = x.to(device)
        outputs, penu, conv_active = model.forward(x)

        return outputs, conv_active

a, b = detect("/home/cjh/peDataSet/Benign/all_benign/7z.exe")
print(f"x {a}, conv {b}")