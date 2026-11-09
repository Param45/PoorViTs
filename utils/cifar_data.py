"""Fast CIFAR-10/100 download + in-memory arrays.

Downloads the dataset as parquet from the Hugging Face CDN (fast), falls back to
an existing torchvision folder (e.g. a Kaggle input dataset) or to the original
Toronto tarball. Everything is cached as one uint8 .npz file:
    x_train (N,32,32,3) uint8, y_train (N,) int64, x_test, y_test
"""
import io
import os
import pickle
import glob
import urllib.request

import numpy as np

HF_URLS = {
    'CIFAR10': ('https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet',
                'img', 'label'),
    'CIFAR100': ('https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{split}-00000-of-00001.parquet',
                 'img', 'fine_label'),
}
TV_FOLDERS = {'CIFAR10': 'cifar-10-batches-py', 'CIFAR100': 'cifar-100-python'}


def npz_path(root, dataset):
    return os.path.join(root, f'{dataset.lower()}_uint8.npz')


def _download(url, dst):
    tmp = dst + '.part'
    print(f'Downloading {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'poorvits-data/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def _from_hf(root, dataset):
    import pyarrow.parquet as pq
    from PIL import Image
    url, img_col, label_col = HF_URLS[dataset]
    out = {}
    for split in ('train', 'test'):
        dst = os.path.join(root, f'{dataset.lower()}_{split}.parquet')
        if not os.path.exists(dst):
            _download(url.format(split=split), dst)
        table = pq.read_table(dst, columns=[img_col, label_col]).to_pydict()
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB'))
                         for d in table[img_col]])
        out[f'x_{split}'] = imgs.astype(np.uint8)
        out[f'y_{split}'] = np.asarray(table[label_col], dtype=np.int64)
    return out


def _from_torchvision_folder(folder, dataset):
    def load(name):
        with open(os.path.join(folder, name), 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        x = np.asarray(d['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.asarray(d['labels'] if 'labels' in d else d['fine_labels'], dtype=np.int64)
        return x, y
    if dataset == 'CIFAR10':
        parts = [load(f'data_batch_{i}') for i in range(1, 6)]
        xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        xte, yte = load('test_batch')
    else:
        (xtr, ytr), (xte, yte) = load('train'), load('test')
    return dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)


def _from_torchvision_download(root, dataset):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == 'CIFAR10' else torchvision.datasets.CIFAR100
    tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
    return dict(x_train=tr.data, y_train=np.asarray(tr.targets, dtype=np.int64),
                x_test=te.data, y_test=np.asarray(te.targets, dtype=np.int64))


def prepare_cifar(root, dataset='CIFAR10'):
    """Make sure `<root>/<dataset>_uint8.npz` exists and return its path."""
    assert dataset in HF_URLS, dataset
    os.makedirs(root, exist_ok=True)
    path = npz_path(root, dataset)
    if os.path.exists(path):
        return path

    data = None
    # 1) an already extracted torchvision folder (root or any Kaggle input dataset)
    candidates = [os.path.join(root, TV_FOLDERS[dataset])]
    candidates += glob.glob(f'/kaggle/input/**/{TV_FOLDERS[dataset]}', recursive=True)
    for c in candidates:
        if os.path.isdir(c):
            try:
                data = _from_torchvision_folder(c, dataset)
                print(f'Loaded {dataset} from {c}')
                break
            except Exception as e:  # corrupted / partial folder
                print(f'Could not read {c}: {e}')
    # 2) Hugging Face CDN parquet (fast)
    if data is None:
        try:
            data = _from_hf(root, dataset)
            print(f'Loaded {dataset} from Hugging Face mirror')
        except Exception as e:
            print(f'Hugging Face download failed ({e}); falling back to torchvision')
    # 3) original server
    if data is None:
        data = _from_torchvision_download(root, dataset)

    n_train, n_test = len(data['y_train']), len(data['y_test'])
    assert data['x_train'].shape == (n_train, 32, 32, 3) and data['x_test'].shape == (n_test, 32, 32, 3)
    np.savez(path + '.tmp.npz', **data)
    os.replace(path + '.tmp.npz', path)
    print(f'Saved {path}: train {n_train}, test {n_test}')
    return path


def load_cifar(root, dataset='CIFAR10'):
    d = np.load(prepare_cifar(root, dataset))
    return d['x_train'], d['y_train'], d['x_test'], d['y_test']


class ArrayDataset:
    """torch Dataset over uint8 HWC arrays, returning PIL images to `transform`."""

    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.x[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--datapath', default='./data')
    p.add_argument('--dataset', default='CIFAR10', choices=list(HF_URLS))
    a = p.parse_args()
    xtr, ytr, xte, yte = load_cifar(a.datapath, a.dataset)
    print(xtr.shape, ytr.shape, xte.shape, yte.shape, np.bincount(ytr))
