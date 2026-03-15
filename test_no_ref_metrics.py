import os
import glob
import torch
import numpy as np
import pyiqa
from pyiqa.utils.img_util import imread2tensor
import csv

# ================================
# 配置
# ================================
img_dir = 'LOL-Blur-Real-Attack/images/output/*.png'
print(img_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================
# 加载指标模型
# ================================
niqe_metric = pyiqa.create_metric('niqe', device=device)
brisque_metric = pyiqa.create_metric('brisque', device=device)
pi_metric = pyiqa.create_metric('pi', device=device)

# ================================
# 遍历图像
# ================================
img_list = sorted(glob.glob(img_dir))
print(f'Found {len(img_list)} images.')

niqe_scores, brisque_scores, pi_scores = [], [], []

for idx, img_path in enumerate(img_list):
    img = imread2tensor(img_path).unsqueeze(0).to(device)
    with torch.no_grad():
        niqe = niqe_metric(img).item()
        brisque = brisque_metric(img).item()
        pi = pi_metric(img).item()

    niqe_scores.append(niqe)
    brisque_scores.append(brisque)
    pi_scores.append(pi)

    print(f"[{idx + 1:03d}/{len(img_list)}] {os.path.basename(img_path)}  "
          f"NIQE: {niqe:.3f}  BRISQUE: {brisque:.3f}  PI: {pi:.3f}")

# ================================
# 计算平均结果
# ================================
niqe_mean = np.mean(niqe_scores)
brisque_mean = np.mean(brisque_scores)
pi_mean = np.mean(pi_scores)

print("\n=========== Overall Results ===========")
print(f"NIQE ↓ : {niqe_mean:.3f}")
print(f"BRISQUE ↓ : {brisque_mean:.3f}")
print(f"PI ↓ : {pi_mean:.3f}")
print("=======================================")

# ================================
# 保存到 CSV
# ================================
csv_path = 'no_ref_metrics.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'NIQE', 'BRISQUE', 'PI'])
    for path, n, b, p in zip(img_list, niqe_scores, brisque_scores, pi_scores):
        writer.writerow([os.path.basename(path), f"{n:.3f}", f"{b:.3f}", f"{p:.3f}"])

    writer.writerow([])
    writer.writerow(['Average', f"{niqe_mean:.3f}", f"{brisque_mean:.3f}", f"{pi_mean:.3f}"])

print(f"\nSaved results to {csv_path}")

