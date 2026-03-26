import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset_path = '/datasets'
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200

lr = 1e-3
epochs = 30
