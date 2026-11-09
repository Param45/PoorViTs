"""Optional: re-test saved checkpoints on the CIFAR-10 test set (10,000 images).

Not needed after run_experiments.py / finetune.py: training already reports the
test top-1 of the final epoch (final_acc) and of best.pth (best_acc) in results.json.
Use this only to re-check checkpoints, e.g. after downloading them.

    python evaluate.py --output_dir save_finetuned --datapath ./data

For each <output_dir>/<run>/best.pth it rebuilds the model from the run's
results.json (same gin bindings / flags) and writes <output_dir>/test_results.csv.
"""
import argparse
import csv
import json
import os
from types import SimpleNamespace

import gin
import torch

from models.build_model import create_model
from utils.cifar_data import load_cifar

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


@torch.inference_mode()
def evaluate(model, x, y, device, bs=1000):
    """Top-1 accuracy (%)."""
    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].permute(0, 3, 1, 2).float() / 255.0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            correct += (model((xb - mean) / std).argmax(1) == y[i:i + bs]).sum()
    return correct.item() * 100.0 / len(y)


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
        acc = evaluate(model, x_test, y_test, device)
        rows.append({'experiment': run, 'test_acc': round(acc, 2)})
        print(f'{run:18s} test top-1 {acc:6.2f}%', flush=True)

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
