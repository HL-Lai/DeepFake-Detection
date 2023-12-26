import os
import cv2
import sys
import time
import joblib
import shutil
import random
import logging
import platform
import numpy as np
from sklearn import *
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from itertools import combinations
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, Subset # "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torchvision.datasets import DatasetFolder
import warnings
warnings.filterwarnings("ignore")

# model = models.resnet50(pretrained=True)
model = models.efficientnet_b4()

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=f'{model._get_name().lower()}.log', mode='a')
formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Download and extract data
from helper import get_data
train_image_path, train_Y, val_image_path, val_Y = get_data(model=model)

lrange = range(5)
i_combs = []
for i in range(1, 5+1):
    i_combs.extend(combinations(lrange, i))

# Modelling with all combinations
from helper import modelling

for i_comb in i_combs:
    modelling(model, i_comb=i_comb, train_image_path=train_image_path, train_Y=train_Y, val_image_path=val_image_path, val_Y=val_Y)
    print(f"{'='*35}")

# Or simply model with any one, e.g.
# modelling(model, i_comb=(0,1)) # Fake-Manipulation-1 + Fake-Manipulation-2