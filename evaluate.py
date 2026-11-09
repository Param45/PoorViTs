"""Test every trained run on the CIFAR-10 test set (10,000 images).

    python evaluate.py --output_dir save_finetuned --datapath ./data

For each <output_dir>/<run>/best.pth it rebuilds the model from the run's
results.json (same gin bindings / flags), evaluates it, and writes
<output_dir>/test_results.csv (overall + per-class top-1 accuracy).
"""
import argparse
import csv
import json
import os
from types import SimpleNamespace

import gin
import numpy as np
import torch

from models.build_model import create_model
from utils.cifar_data import load_cifar

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def evaluate(model, x, y, device, bs=1000):
    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    preds = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            preds.append(model((xb - mean) / std).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    acc = (pred == y).mean() * 100
    per_class = [(pred[y == c] == c).mean() * 100 for c in range(10)]
    return acc, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='save_finetuned')
    p.add_argument('--datapath', default='./data')
    p.add_argument('--checkpoint', default='best.pth', choices=['best.pth', 'checkpoint.pth'])
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, x_test, y_test = load_cifar(a.datapath, 'CIFAR10')
    rows = []
    for run in sorted(os.listdir(a.output_dir)):
        rdir = os.path.join(a.output_dir, run)
        res_p, ckpt_p = os.path.join(rdir, 'results.json'), os.path.join(rdir, a.checkpoint)
        if not (os.path.isfile(res_p) and os.path.isfile(ckpt_p)):
            continue
        with open(res_p) as f:
            res = json.load(f)
        r_args = SimpleNamespace(**res['args'])
        gin.clear_config()
        gin.parse_config(r_args.gin)
        model = create_model(32, 10, r_args).to(device)
        state = torch.load(ckpt_p, map_location=device)
        state = state.get('model_state_dict', state)
        model.load_state_dict(state)
        acc, per_class = evaluate(model, x_test, y_test, device)
        rows.append({'experiment': run, 'test_acc': round(acc, 2),
                     **{c: round(v, 1) for c, v in zip(CLASSES, per_class)}})
        print(f'{run:18s} test top-1 {acc:6.2f}%   ' +
              ' '.join(f'{c[:4]}:{v:4.1f}' for c, v in zip(CLASSES, per_class)), flush=True)

    if rows:
        out = os.path.join(a.output_dir, 'test_results.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'saved {out}')
    else:
        print(f'no finished runs found in {a.output_dir}')


if __name__ == '__main__':
    main()
