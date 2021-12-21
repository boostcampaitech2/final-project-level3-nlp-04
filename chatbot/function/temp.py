import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

a = torch.rand((3, 256, 256))
x = a.numpy()

x = np.transpose(x, axes=(1, 2, 0))
y = np.random.randint(255, size=(400, 400, 3),dtype=np.uint8)

print(x.shape)
print(y.shape)
x = x.astype(np.uint8)
Image.fromarray(x)
# Image.fromarray(y)