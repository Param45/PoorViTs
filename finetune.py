"""
Train a Tiny ViT on CIFAR-10 (Powerful Design of Small Vision Transformer on CIFAR10).

single GPU:
    python finetune.py --datapath ./data --amp --exp_name baseline
two GPUs (DDP, --batch_size is per GPU):
    torchrun --nproc_per_node=2 finetune.py --datapath ./data --amp --batch_size 128 --exp_name ddp_bs256
ablations (see run_experiments.py for the full list):
    --aa / --mu / --cm            disable AutoAugment / Mixup / CutMix
    --sin_pos, --whitening        patch-token initialisation
    --gin 'Attention.latent_mode="q"'           MLA low-rank compression
    --gin ViT.embed_dim=96 ViT.num_cls_token=2  multi-CLS tokens
"""
import argparse
import json
import logging as log
import math
import os
import random
import time
import warnings

import gin
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style

from models.build_model import create_model
from utils.dataloader import datainfo, dataload
from utils.logger_dict import Logger_dict
from utils.losses import LabelSmoothingCrossEntropy
from utils.mix import cutmix_data, mixup_data, mixup_criterion
from utils.sampler import RASampler
from utils.scheduler import build_scheduler

warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vit', 'swin', 'cait', 'none']

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    n_patterns = eigenvectors_scaled.shape[0]
    layer.weight.data[:n_patterns] = eigenvectors_scaled.to(layer.weight.dtype)

def init_parser():
    parser = argparse.ArgumentParser(
        description='Vit small datasets quick training script')

    parser.add_argument('--profile', action='store_true', default=False, help='profile training')
    parser.add_argument('--whitening', action='store_true',default=False, help='Use whitening initialization')
    parser.add_argument('--sin_pos', action='store_true',default=False, help='Use sin position embedding')
    parser.add_argument('--gin', nargs='+', type=str, default=[], help='Configure Modules')
    # Data args
    parser.add_argument('--datapath', default='./data',
                        type=str, help='dataset path')

    parser.add_argument('--dataset', default='CIFAR10', choices=[
                        'CIFAR10', 'CIFAR100', 'Tiny-Imagenet', 'SVHN', 'CINIC'], type=str, help='small dataset path')

    parser.add_argument('--data_pipeline', default='gpu', choices=['gpu', 'cpu'],
                        help='gpu: augment on the GPU (CIFAR only, fast on few-CPU machines such as Kaggle); '
                             'cpu: original torchvision/PIL DataLoader pipeline')

    parser.add_argument('-j', '--workers', default=min(10, os.cpu_count() or 1), type=int,
                        metavar='N', help='number of data loading workers (cpu pipeline)')

    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')

    parser.add_argument('--warmup', default=10, type=int,
                        metavar='N', help='number of warmup epochs')

    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size per GPU (default: 256)', dest='batch_size')

    parser.add_argument('--opt', default='adamw', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.002, type=float,
                        help='initial learning rate (paper: 0.002)')

    parser.add_argument('--weight-decay', default=5e-2,
                        type=float, help='weight decay (default: 5e-2)')

    parser.add_argument('--arch', type=str, default='vit', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--amp', action='store_true',
                        help='enable mixed precision')

    parser.add_argument('--compile', action='store_true',
                        help='torch.compile the model (faster training)')

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--exp_name', type=str, default='',
                        help='name of the output folder (default: derived from args)')

    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'save_finetuned'),
                        help='where runs are stored')

    parser.add_argument('--eval_freq', type=int, default=5,
                        help='validate every N epochs (and always after the last one)')

    parser.add_argument('--benchmark_steps', type=int, default=0,
                        help='if > 0: only measure training throughput over this many steps and exit')

    parser.add_argument('--wandb', action='store_true', help='log to Weights & Biases')

    parser.add_argument('--train_subset', type=int, default=0,
                        help='debug: only use the first N train/test images (CIFAR)')

    parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--sd', default=0.1, type=float,
                        help='rate of stochastic depth')

    parser.add_argument('--resume', default=False, help='Version')

    parser.add_argument('--aa', action='store_false',
                        help='Auto augmentation used'),

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--cm', action='store_false', help='Use Cutmix')

    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')

    parser.add_argument('--mu', action='store_false', help='Use Mixup')

    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')

    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')

    parser.add_argument('--ra', type=int, default=3,
                        help='repeated augmentation')

    parser.add_argument('--re', default=0.25, type=float,
                        help='Random Erasing probability')

    parser.add_argument('--re_sh', default=0.4, type=float,
                        help='max erasing area')

    parser.add_argument('--re_r1', default=0.3, type=float,
                        help='aspect of erasing area')

    parser.add_argument('--is_LSA', action='store_true',
                        help='Locality Self-Attention')

    parser.add_argument('--is_SPT', action='store_true',
                        help='Shifted Patch Tokenization')

    parser.add_argument('--pretrained_weights', default='',
                        type=str, help="Path to pretrained weights to evaluate.")

    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--patch_size', default=4,
                        type=int, help='patch size for ViT')

    parser.add_argument('--vit_mlp_ratio', default=2, type=int,
                        help='MLP layers in the transformer encoder')

    return parser


def is_main():
    return (not args.distributed) or args.rank == 0


def banner(msg, color=Fore.YELLOW):
    print(color + '*'*80)
    logger.debug(msg)
    print('*'*80 + Style.RESET_ALL)


def autocast(enabled):
    if args.device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=enabled)


def make_grad_scaler(enabled):
    enabled = enabled and args.device.type == 'cuda'
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def synchronize():
    if args.device.type == 'cuda':
        torch.cuda.synchronize()


class BatchAugmentation:
    # https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
    def __init__(self, sampler, repeat=1):
        self.sampler = sampler
        self.repeat = repeat

    def __len__(self):
        return len(self.sampler)*self.repeat

    def __iter__(self):
        for index in self.sampler:
            for _ in range(self.repeat):
                yield index

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)


def build_loaders(args, data_info):
    """Returns (train_loader, val_loader). Train batches may live on CPU or GPU."""
    mean, std = data_info['stat']
    cifar = args.dataset in ('CIFAR10', 'CIFAR100')
    if cifar:
        from utils.cifar_data import load_cifar
        x_train, y_train, x_test, y_test = load_cifar(args.datapath, args.dataset)
        if args.train_subset:
            x_train, y_train = x_train[:args.train_subset], y_train[:args.train_subset]
            x_test, y_test = x_test[:args.train_subset], y_test[:args.train_subset]

    if args.data_pipeline == 'gpu':
        assert cifar, '--data_pipeline gpu supports CIFAR10/CIFAR100 only, use --data_pipeline cpu'
        from utils.gpu_aug import GPULoader
        device = args.device
        banner(f'GPU data pipeline (AutoAugment={args.aa}, RandomErasing={args.re}, RA={args.ra})')
        train_loader = GPULoader(x_train, y_train, args.batch_size, mean, std, device, train=True,
                                 aa=args.aa, re=args.re, re_sh=args.re_sh, re_r1=args.re_r1, ra=args.ra,
                                 distributed=args.distributed, rank=args.rank, world_size=args.world_size,
                                 seed=args.seed)
        val_loader = GPULoader(x_test, y_test, 1000, mean, std, device, train=False)
        return train_loader, val_loader

    '''
        Data Augmentation (original CPU pipeline)
    '''
    normalize = [transforms.Normalize(mean=mean, std=std)]
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]
    if args.aa:
        if 'CIFAR' in args.dataset:
            from utils.autoaug import CIFAR10Policy
            augmentations += [CIFAR10Policy()]
        elif 'SVHN' in args.dataset:
            from utils.autoaug import SVHNPolicy
            augmentations += [SVHNPolicy()]
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [ImageNetPolicy()]
        banner('Autoaugmentation used')
    augmentations += [transforms.ToTensor(), *normalize]
    if args.re > 0:
        from utils.random_erasing import RandomErasing
        augmentations += [RandomErasing(probability=args.re, sh=args.re_sh, r1=args.re_r1, mean=mean)]
        banner(f'Random erasing({args.re}) used ')
    augmentations = transforms.Compose(augmentations)

    if cifar:
        from utils.cifar_data import ArrayDataset
        train_dataset = ArrayDataset(x_train, y_train, augmentations)
        val_dataset = ArrayDataset(x_test, y_test, transforms.Compose([transforms.ToTensor(), *normalize]))
    else:
        train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)

    persistent = args.workers > 0
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=BatchAugmentation(sampler_train, args.ra),
            num_workers=args.workers, batch_size=args.batch_size, drop_last=True, pin_memory=True,
            persistent_workers=persistent)
    else:
        sampler_train = RASampler(
            len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=sampler_train,
            persistent_workers=persistent)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=args.workers)
    return train_loader, val_loader


def set_loader_epoch(loader, epoch):
    if hasattr(loader, 'set_epoch'):
        loader.set_epoch(epoch)
    elif isinstance(getattr(loader, 'sampler', None), BatchAugmentation):
        loader.sampler.set_epoch(epoch)


def main(args):
    global best_acc1

    if args.device.type == 'cuda':
        torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args)

    model = create_model(data_info['img_size'], data_info['n_classes'], args)
    model.to(args.device)
    model_without_ddp = model

    print(Fore.GREEN+'*'*80)
    logger.debug(f"Creating model: {model_name}")
    print(model_without_ddp)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*'*80+Style.RESET_ALL)

    if os.path.isfile(args.pretrained_weights):
        model_dict = model.state_dict()
        print("loading pretrained weights . . .")
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(
                f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k,
                      v in state_dict.items()}
        state_dict = {k: v if v.size() == model_dict[k].size(
        ) else model_dict[k] for k, v in zip(model_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict, strict=False)

    if args.ls:
        banner('label smoothing used')
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    if args.sd > 0.:
        banner(f'Stochastic depth({args.sd}) used ')
    if args.cm:
        banner('Cutmix used')
    if args.mu:
        banner('Mixup used')
    if args.ra > 1:
        banner(f'Repeated Aug({args.ra}) used')

    train_loader, val_loader = build_loaders(args, data_info)

    if args.whitening:
        # done before DDP wrapping so that every rank starts from the same weights (DDP broadcasts rank 0)
        banner('Whitening initialisation of the patch embedding')
        with torch.no_grad():
            train_images = []
            for x, y in train_loader:
                train_images.append(x.to(args.device))
                if len(train_images)*len(x) > 5000:
                    break
            train_images = torch.cat(train_images)
            init_whitening_conv(model_without_ddp.patch_embed.proj, train_images)
            del train_images

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if args.device.type == 'cuda' else None)
    if args.compile:
        banner('torch.compile enabled')
        model = torch.compile(model)

    '''
        Training
    '''

    if args.opt == 'muon':
        from utils.muon import Muon
        # code: https://github.com/KellerJordan/Muon
        muon_params = []
        adamw_params = []
        for name, p in model_without_ddp.named_parameters():
            if 'head' in name or 'fc' in name or 'embed' in name:
                adamw_params.append(p)
            elif p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        print(f"Muon parameters: {len(muon_params)}")
        optimizer = Muon(muon_params, lr=0.02, momentum=0.9,
                         adamw_params=adamw_params, adamw_lr=1e-3, adamw_wd=args.weight_decay)
    elif args.opt == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    loss_scaler = make_grad_scaler(args.amp)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    if args.profile:
        train_profile(train_loader, model, criterion, optimizer, args.epochs, scheduler, loss_scaler, args)
        return

    if args.benchmark_steps > 0:
        benchmark(train_loader, model, criterion, optimizer, scheduler, loss_scaler, args, n_parameters)
        return

    print()
    print("Beginning training")
    print()

    t_start = time.time()
    acc1 = 0.0
    history = []
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        set_loader_epoch(train_loader, epoch)
        lr, t_loss, t_acc = train(train_loader, model, criterion, optimizer,
                                  epoch, scheduler, loss_scaler, args)
        if is_main():
            print(f'[Epoch {epoch+1}/{args.epochs}] train loss {t_loss:.4f}  train acc {t_acc:6.2f}  '
                  f'lr {lr:.2e}  epoch {time.time()-t_epoch:.1f}s  elapsed {time.time()-t_start:.0f}s', flush=True)
        last = epoch == args.epochs - 1
        if (epoch % args.eval_freq == 0 or last) and is_main():
            acc1 = validate(val_loader, model_without_ddp, criterion, lr, args, epoch=epoch)
            logger_dict.print()
            history.append({'epoch': epoch + 1, 'val_acc': acc1, 'train_loss': t_loss, 'train_acc': t_acc,
                            'time': time.time() - t_start})

            if acc1 > best_acc1:
                print('* Best model update *')
                best_acc1 = acc1

                torch.save({
                    'model_state_dict': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))

            print(f'Best acc1 {best_acc1:.2f}')
            print('*'*80)
            print(Style.RESET_ALL)

    runtime = time.time() - t_start
    if is_main():
        print(Fore.RED+'*'*80)
        logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}, runtime: {runtime:.0f}s')
        print('*'*80+Style.RESET_ALL)
        torch.save(model_without_ddp.state_dict(), os.path.join(save_path, 'checkpoint.pth'))
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump({'exp_name': args.exp_name or model_name, 'best_acc': best_acc1, 'final_acc': acc1,
                       'runtime_sec': runtime, 'epochs': args.epochs, 'params': n_parameters,
                       'world_size': args.world_size, 'global_batch_size': args.batch_size * args.world_size,
                       'history': history, 'args': {k: v for k, v in vars(args).items()
                                                    if isinstance(v, (int, float, str, bool, list))}},
                      f, indent=2)


def mix_batch(images, target, args):
    """CutMix / Mixup exactly as in the original training loop.
    Returns (images, y_a, y_b, lam) or (images, None, None, None) when not mixed."""
    if not (args.cm or args.mu) or np.random.rand(1) >= args.mix_prob:
        return images, None, None, None
    use_cutmix = args.cm and (not args.mu or np.random.rand(1) < 0.5)
    if use_cutmix:
        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
        images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
    else:
        images, y_a, y_b, lam = mixup_data(images, target, args)
    return images, y_a, y_b, lam


def train_step(images, target, model, criterion, optimizer, scheduler, loss_scaler, args):
    if images.device != args.device:
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
    images, y_a, y_b, lam = mix_batch(images, target, args)
    with autocast(args.amp):
        output = model(images)
        if lam is not None:
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            loss = criterion(output, target)

    optimizer.zero_grad(set_to_none=True)
    loss_scaler.scale(loss).backward()
    loss_scaler.step(optimizer)
    loss_scaler.update()
    scheduler.step()  # change learning rate (per iteration)
    return loss.detach(), output.detach(), target


def train(train_loader, model, criterion, optimizer, epoch, scheduler, loss_scaler, args):
    model.train()
    loss_sum = torch.zeros((), device=args.device)
    correct = torch.zeros((), device=args.device)
    n = 0
    for i, (images, target) in enumerate(train_loader):
        loss, output, target = train_step(images, target, model, criterion, optimizer, scheduler,
                                          loss_scaler, args)
        n += target.size(0)
        loss_sum += loss.float() * target.size(0)
        correct += (output.argmax(1) == target).sum()
        if args.wandb and wandb.run:
            wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    lr = optimizer.param_groups[0]["lr"]
    avg_loss, avg_acc1 = loss_sum.item() / n, correct.item() * 100.0 / n
    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    if args.wandb and wandb.run:
        wandb.log({"train/loss": avg_loss, "train/acc": avg_acc1, 'epoch': epoch})
    return lr, avg_loss, avg_acc1


def benchmark(train_loader, model, criterion, optimizer, scheduler, loss_scaler, args, n_parameters,
              warmup=25):
    """Measure training throughput (images/s, summed over all GPUs) including the data pipeline."""
    model.train()
    done, t0 = 0, None
    t_compile = time.time()
    while True:
        for images, target in train_loader:
            if done == warmup:
                synchronize()
                t0 = time.time()
                print(f'warmup (incl. compile) took {t0 - t_compile:.1f}s', flush=True)
            train_step(images, target, model, criterion, optimizer, scheduler, loss_scaler, args)
            done += 1
            if done == warmup + args.benchmark_steps:
                break
        if done == warmup + args.benchmark_steps:
            break
    synchronize()
    elapsed = time.time() - t0
    img_s = args.benchmark_steps * args.batch_size * args.world_size / elapsed
    if is_main():
        res = {'img_per_sec': img_s, 'steps': args.benchmark_steps, 'global_batch_size':
               args.batch_size * args.world_size, 'compile_warmup_sec': t0 - t_compile,
               'params': n_parameters, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9 if args.device.type == 'cuda' else 0.0}
        print('BENCHMARK ' + json.dumps(res), flush=True)
        with open(os.path.join(save_path, 'benchmark.json'), 'w') as f:
            json.dump(res, f)


def train_profile(train_loader, model, criterion, optimizer, epoch, scheduler, loss_scaler, args):
    import torch.profiler
    model.train()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(save_path, 'profile')),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, (images, target) in enumerate(train_loader):
            with torch.profiler.record_function("train_step"):
                train_step(images, target, model, criterion, optimizer, scheduler, loss_scaler, args)
            prof.step()
            if i > 10:
                break


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_sum = torch.zeros((), device=args.device)
    correct = torch.zeros((), device=args.device)
    n = 0
    with torch.no_grad(), autocast(args.amp):
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            output = model(images)
            loss = criterion(output.float(), target)
            n += images.size(0)
            loss_sum += loss.float() * images.size(0)
            correct += (output.argmax(1) == target).sum()
    avg_loss, avg_acc1 = loss_sum.item() / n, correct.item() * 100.0 / n
    print(Fore.BLUE)
    print(f'[Epoch {epoch+1}][V]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print('*'*80)

    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)

    if args.wandb and wandb.run:
        wandb.log({"val/loss": avg_loss, "val/acc": avg_acc1, 'epoch': epoch})

    return avg_acc1


class _NullLoggerDict:
    def update(self, *a, **k):
        pass

    def print(self):
        pass


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    gin.parse_config(args.gin)

    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        if use_cuda:
            torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://",
                                world_size=args.world_size, rank=args.rank)
        args.distributed = True
        dist.barrier()
        import builtins as __builtin__
        builtin_print = __builtin__.print
        ismaster = args.rank == 0

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if ismaster or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device('cuda', args.gpu) if use_cuda else torch.device('cpu')

    # random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.benchmark = True

    model_name = args.arch

    if not args.is_SPT:
        model_name += "-Base"
    else:
        print("spt present")
        model_name += "-SPT"

    if args.is_LSA:
        print("lsa present")
        model_name += "-LSA"

    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(args.output_dir, args.exp_name or model_name)
    os.makedirs(save_path, exist_ok=True)

    wandb = None
    if args.wandb and is_main():
        import wandb
        wandb.init(job_type='finetune', dir=save_path, name=args.exp_name or model_name,
                   config=args.__dict__)

    # logger
    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    if is_main():
        fileHandler = log.FileHandler(log_dir, 'a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.setLevel(level=log.DEBUG)
    else:
        logger.setLevel(level=log.WARNING)

    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    logger_dict = Logger_dict(logger, save_path) if is_main() else _NullLoggerDict()

    main(args)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
