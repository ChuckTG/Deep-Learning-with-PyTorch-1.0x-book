import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder


transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.12,0.11,0.40],[0.89,0.21,0.12])])

train = ImageFolder('Data/Train_Data/train/',transform)
valid = ImageFolder('Data/Train_Data/valid/',transform)


import matplotlib.pyplot as plt

def imshow(img):
    img  = img.numpy().transpose(1,2,0)
    mean = np.array([0.12,0.11,0.40])
    std  = np.array([0.89,0.21,0.12])
    img  = std * img + mean
    img  = np.clip(img,0,1)
    plt.imshow(img)
    plt.show()

imshow(train[0][0])
