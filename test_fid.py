import os
import csv
import numpy as np
import torch
import pyiqa
import argparse
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS
import glob
device = torch.device('cuda:0')
fid_metric = pyiqa.create_metric('fid', device=device)
f1='test_gopro/images/GT/'
f2='test_gopro/images/output/'

FID = fid_metric(f1,f2)
print(f1,f2,FID)