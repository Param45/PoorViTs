"""Run the experiments of "Powerful Design of Small Vision Transformer on CIFAR10"
(Tables 2 and 3) within a fixed time budget, e.g. on Kaggle's 2x T4.

    python run_experiments.py --budget_hours 3.6

1. downloads CIFAR-10 (Hugging Face CDN, falls back to the original server)
2. benchmarks training throughput (eager vs torch.compile, single GPU and DDP)
3. picks the largest number of epochs (<= 100, the paper's value) such that all
   selected experiments finish inside the budget; warmup stays 10% of training
4. runs single-GPU experiments in parallel (one per GPU), then the DDP ones on
   all GPUs; finished runs are skipped when the script is restarted
5. writes summary.md / summary.csv / curves.png in --output_dir

Metrics are kept to what the paper reports: test top-1 (best + final epoch,
evaluated every --eval_freq epochs), runtime, and the mean train loss per epoch
(accumulated on the GPU, one sync per epoch) to spot diverging runs.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SIZE = 50000


def experiments(n_gpus):
    """name, extra args, 'single'|'ddp', paper val acc, paper runtime (s), description"""
    g = max(1, n_gpus)
    mla = lambda m: ['--gin', f'Attention.latent_mode="{m}"']
    return [
        ('baseline', [], 'single', 93.65, 4658, 'Baseline: AA + Mixup + CutMix + RA3 + LS + SD + RE'),
        ('no_aa', ['--aa'], 'single', 92.14, 4684, 'Augmentation: without AutoAugment'),
        ('no_mixup', ['--mu'], 'single', 93.68, 4656, 'Augmentation: without Mixup'),
        ('no_cutmix', ['--cm'], 'single', 92.77, 4651, 'Augmentation: without CutMix'),
        ('sin_pos', ['--sin_pos'], 'single', 93.43, 4687, 'Patch: fixed 2D sin-cos position embedding'),
        ('whitening', ['--whitening'], 'single', 93.28, 4676, 'Patch: whitening init of patch embedding'),
        ('mla_q', mla('q'), 'single', 93.32, 4716, 'MLA: low-rank queries (d_c=64)'),
        ('mla_k', mla('k'), 'single', 92.50, 4711, 'MLA: low-rank keys'),
        ('mla_kv', mla('kv'), 'single', 92.43, 4756, 'MLA: low-rank keys + values'),
        ('mla_qk', mla('qk'), 'single', 92.42, 4785, 'MLA: low-rank queries + keys'),
        ('mla_qkv', mla('qkv'), 'single', 91.80, 4805, 'MLA: low-rank queries + keys + values'),
        ('dim96_cls1', ['--gin', 'ViT.embed_dim=96'], 'single', 90.41, None, 'Table 3: dim 96, 1 CLS token'),
        ('dim96_cls2', ['--gin', 'ViT.embed_dim=96', 'ViT.num_cls_token=2'], 'single', 93.95, None,
         'Table 3: dim 96, 2 CLS tokens'),
        ('ddp_bs256', ['--batch_size', str(256 // g)], 'ddp', 93.44, 3582, f'DDP {g} GPUs, global batch 256'),
        ('ddp_bs1024', ['--batch_size', str(1024 // g)], 'ddp', 92.09, 1371, f'DDP {g} GPUs, global batch 1024'),
        ('ddp_lion_bs1024', ['--batch_size', str(1024 // g), '--opt', 'lion', '--lr', '2e-4',
                             '--weight-decay', '0.5'], 'ddp', 92.38, 1545,
         f'DDP {g} GPUs, global batch 1024, Lion (lr 2e-4, wd 0.5)'),
    ]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fmt_t(sec):
    return f'{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m'


class Runner:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.deadline = self.t0 + a.budget_hours * 3600
        self.port = 29500
        os.makedirs(a.output_dir, exist_ok=True)

    # -------------------------------------------------------------- process
    def cmd(self, name, extra, n_proc, epochs, compile_, bench_steps=0):
        common = ['finetune.py', '--datapath', self.a.datapath, '--output_dir', self.a.output_dir,
                  '--exp_name', name, '--amp', '--epochs', str(epochs),
                  '--warmup', str(max(1, round(epochs * 0.1))),
                  '--eval_freq', str(self.a.eval_freq), '--seed', str(self.a.seed)]
        if compile_:
            common.append('--compile')
        if bench_steps:
            common += ['--benchmark_steps', str(bench_steps)]
        if n_proc > 1:
            self.port += 1
            return [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={n_proc}',
                    f'--master_port={self.port}'] + common + extra
        return [sys.executable] + common + extra

    def launch(self, name, cmd, gpus):
        os.makedirs(os.path.join(self.a.output_dir, name), exist_ok=True)
        logf = open(os.path.join(self.a.output_dir, name, 'train.log'), 'w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1')
        if len(gpus) > 1:
            env.setdefault('NCCL_P2P_DISABLE', '1')  # P2P between the T4s of a Kaggle VM can hang NCCL
        logf.write(' '.join(cmd) + '\n\n')
        logf.flush()
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT), logf

    def tail(self, name, pattern=None, n=1):
        try:
            with open(os.path.join(self.a.output_dir, name, 'train.log'), errors='ignore') as f:
                lines = [l.rstrip() for l in f if l.strip() and (pattern is None or pattern in l)]
            return '\n'.join(lines[-n:])
        except OSError:
            return ''

    # ------------------------------------------------------------ benchmark
    def bench(self, name, extra, gpus, compile_):
        p, f = self.launch(name, self.cmd(name, extra, len(gpus), 100, compile_, self.a.bench_steps), gpus)
        return p, f

    def read_bench(self, name, proc, logf):
        proc.wait()
        logf.close()
        path = os.path.join(self.a.output_dir, name, 'benchmark.json')
        if proc.returncode != 0 or not os.path.exists(path):
            log(f'benchmark {name} FAILED (exit {proc.returncode}):\n{self.tail(name, n=15)}')
            return None
        with open(path) as fh:
            r = json.load(fh)
        log(f'benchmark {name}: {r["img_per_sec"]:.0f} img/s, warmup/compile {r["compile_warmup_sec"]:.0f}s, '
            f'peak mem {r["peak_mem_gb"]:.2f} GB')
        return r

    def benchmark(self, n_gpus, need_ddp):
        bdir = '_bench'
        jobs = {}
        if self.a.compile == 'auto' and n_gpus >= 2:
            jobs['eager'] = self.bench(f'{bdir}/eager', [], [0], False)
            jobs['compiled'] = self.bench(f'{bdir}/compiled', [], [1], True)
        else:
            c = self.a.compile != 'off'
            jobs['compiled' if c else 'eager'] = self.bench(f'{bdir}/{"compiled" if c else "eager"}', [], [0], c)
        res = {k: self.read_bench(f'{bdir}/{k}', *v) for k, v in jobs.items()}
        if res.get('compiled') is None and res.get('eager') is None:
            if 'eager' in res:
                raise SystemExit('Benchmark failed, see the log above.')
            log('torch.compile failed -> retrying in eager mode')
            res['eager'] = self.read_bench(f'{bdir}/eager', *self.bench(f'{bdir}/eager', [], [0], False))
            if res['eager'] is None:
                raise SystemExit('Benchmark failed in eager mode too, see the log above.')
        e, c = res.get('eager'), res.get('compiled')
        # compiled wins if it is >10% faster (its compile time per run is paid separately below)
        use_compile = c is not None and (e is None or c['img_per_sec'] > 1.1 * e['img_per_sec'])
        single = c if use_compile else e
        log(f'using {"torch.compile" if use_compile else "eager mode"} '
            f'({single["img_per_sec"]:.0f} img/s per GPU)')
        ddp = None
        if need_ddp:
            name = f'{bdir}/ddp'
            ddp = self.read_bench(name, *self.bench(name, ['--batch_size', str(256 // n_gpus)],
                                                    list(range(n_gpus)), use_compile))
        return use_compile, single, ddp

    # ------------------------------------------------------------- planning
    def plan_epochs(self, exps, n_gpus, single, ddp):
        spe = TRAIN_SIZE * 3  # samples per epoch with repeated augmentation 3
        n_single = sum(1 for e in exps if e[2] == 'single')
        n_ddp = sum(1 for e in exps if e[2] == 'ddp')
        rounds = math.ceil(n_single / max(1, n_gpus))
        ovh_s = single['compile_warmup_sec'] + 45  # process start, data to GPU, compile, evals, saving
        ovh_d = (ddp['compile_warmup_sec'] + 60) if ddp else 0
        avail = (self.deadline - time.time()) * 0.93  # keep a safety margin
        per_epoch = rounds * spe / single['img_per_sec'] + (n_ddp * spe / ddp['img_per_sec'] if ddp else 0)
        fixed = rounds * ovh_s + n_ddp * ovh_d
        epochs = int((avail - fixed) / per_epoch)
        log(f'{n_single} single-GPU runs in {rounds} rounds on {n_gpus} GPU(s) + {n_ddp} DDP runs; '
            f'{fmt_t(avail)} usable -> {epochs} epochs fit')
        return max(self.a.min_epochs, min(100, epochs))

    def est_run_sec(self, kind, epochs, single, ddp):
        r = single if kind == 'single' else ddp
        return epochs * TRAIN_SIZE * 3 / r['img_per_sec'] + r['compile_warmup_sec'] + 45

    # ------------------------------------------------------------ execution
    def done(self, name, epochs):
        p = os.path.join(self.a.output_dir, name, 'results.json')
        if not os.path.exists(p):
            return False
        with open(p) as f:
            return json.load(f).get('epochs') == epochs

    def run_all(self, exps, n_gpus, epochs, compile_, single, ddp):
        status = {}
        todo_single = Queue()
        for e in exps:
            if self.done(e[0], epochs):
                status[e[0]] = 'done (cached)'
            elif e[2] == 'single':
                todo_single.put(e)
        running = {}  # gpu -> (name, proc, logf, t_start)
        last_report = time.time()

        def start(e, gpus):
            est = self.est_run_sec(e[2], epochs, single, ddp)
            if time.time() + est > self.deadline:
                status[e[0]] = f'skipped (needs ~{fmt_t(est)}, budget left {fmt_t(self.deadline - time.time())})'
                log(f'{e[0]}: {status[e[0]]}')
                return None
            proc, logf = self.launch(e[0], self.cmd(e[0], e[1], len(gpus), epochs, compile_), gpus)
            log(f'START {e[0]} on GPU {gpus} (~{fmt_t(est)}): {e[5]}')
            return (e[0], proc, logf, time.time())

        def reap(block=False):
            nonlocal last_report
            while True:
                for gpu, (name, proc, logf, ts) in list(running.items()):
                    if proc.poll() is not None:
                        logf.close()
                        del running[gpu]
                        p = os.path.join(self.a.output_dir, name, 'results.json')
                        if proc.returncode == 0 and os.path.exists(p):
                            with open(p) as f:
                                r = json.load(f)
                            status[name] = 'done'
                            log(f'DONE  {name}: best {r["best_acc"]:.2f}%  final {r["final_acc"]:.2f}%  '
                                f'({fmt_t(time.time() - ts)})')
                        else:
                            status[name] = f'FAILED (exit {proc.returncode})'
                            log(f'FAIL  {name}:\n{self.tail(name, n=20)}')
                if time.time() - last_report > self.a.report_every:
                    last_report = time.time()
                    for gpu, (name, *_rest) in running.items():
                        log(f'  {name}: {self.tail(name, pattern="[Epoch")}')
                    log(f'  elapsed {fmt_t(time.time() - self.t0)}, budget left {fmt_t(self.deadline - time.time())}')
                if not block or not running:
                    return
                time.sleep(5)

        # single-GPU experiments: one per GPU, in parallel
        while True:
            reap()
            free = [g for g in range(n_gpus) if g not in running]
            if free:
                try:
                    e = todo_single.get_nowait()
                except Empty:
                    if not running:
                        break
                    time.sleep(5)
                    continue
                job = start(e, [free[0]])
                if job:
                    running[free[0]] = job
            else:
                time.sleep(5)

        # DDP experiments: all GPUs
        for e in exps:
            if e[2] != 'ddp' or e[0] in status:
                continue
            if n_gpus < 2:
                status[e[0]] = 'skipped (needs 2+ GPUs)'
                continue
            job = start(e, list(range(n_gpus)))
            if job:
                running['ddp'] = job
                reap(block=True)
        return status

    # --------------------------------------------------------------- report
    def report(self, exps, status, epochs, extra_info):
        rows = []
        for name, _, kind, p_acc, p_rt, desc in exps:
            path = os.path.join(self.a.output_dir, name, 'results.json')
            r = json.load(open(path)) if os.path.exists(path) else None
            rows.append({'experiment': name, 'description': desc, 'paper_acc': p_acc,
                         'best_acc': round(r['best_acc'], 2) if r else None,
                         'final_acc': round(r['final_acc'], 2) if r else None,
                         'delta_vs_baseline': None, 'paper_delta_vs_baseline': round(p_acc - 93.65, 2),
                         'epochs': r['epochs'] if r else None,
                         'runtime_sec': round(r['runtime_sec']) if r else None, 'paper_runtime_sec': p_rt,
                         'params': r['params'] if r else None, 'status': status.get(name, 'not run'),
                         'history': r['history'] if r else []})
        base = next((r['best_acc'] for r in rows if r['experiment'] == 'baseline'), None)
        for r in rows:
            if base is not None and r['best_acc'] is not None:
                r['delta_vs_baseline'] = round(r['best_acc'] - base, 2)

        cols = ['experiment', 'description', 'paper_acc', 'best_acc', 'final_acc', 'paper_delta_vs_baseline',
                'delta_vs_baseline', 'epochs', 'runtime_sec', 'paper_runtime_sec', 'params', 'status']
        with open(os.path.join(self.a.output_dir, 'summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        md = ['# Results: Powerful Design of Small Vision Transformer on CIFAR10', '',
              extra_info, '',
              '| Experiment | Description | Paper acc | Our best acc | Our final acc | Paper diff vs base | Our diff vs base '
              '| Epochs | Runtime (s) | Paper runtime (s) | Params | Status |',
              '|' + '---|' * 12]
        for r in rows:
            md.append('| ' + ' | '.join('' if r[c] is None else str(r[c]) for c in cols) + ' |')
        md.append('')
        md.append('Paper numbers: 100 epochs on one RTX 3090 (DDP rows: 4x RTX 3090). '
                  'diff = accuracy minus the baseline of the same table.')
        with open(os.path.join(self.a.output_dir, 'summary.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(md) + '\n')
        print('\n'.join(md), flush=True)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for r in rows:
                if r['history']:
                    ep = [h['epoch'] for h in r['history']]
                    axes[0].plot(ep, [h['val_acc'] for h in r['history']], label=r['experiment'])
            axes[0].set_xlabel('epoch')
            axes[0].set_ylabel('val top-1 (%)')
            axes[0].set_ylim(30, 100)
            axes[0].grid(alpha=.3)
            axes[0].legend(fontsize=8, ncol=2)
            done = [r for r in rows if r['best_acc'] is not None]
            x = range(len(done))
            axes[1].bar([i - 0.2 for i in x], [r['paper_acc'] for r in done], 0.4, label='paper (100 ep)')
            axes[1].bar([i + 0.2 for i in x], [r['best_acc'] for r in done], 0.4, label=f'ours ({epochs} ep)')
            axes[1].set_xticks(list(x))
            axes[1].set_xticklabels([r['experiment'] for r in done], rotation=60, ha='right', fontsize=8)
            lo = min([r['best_acc'] for r in done] + [r['paper_acc'] for r in done]) - 2
            axes[1].set_ylim(lo, 96)
            axes[1].legend()
            axes[1].grid(alpha=.3, axis='y')
            fig.tight_layout()
            fig.savefig(os.path.join(self.a.output_dir, 'curves.png'), dpi=120)
        except Exception as ex:  # plotting is optional
            log(f'plotting skipped: {ex}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datapath', default=os.path.join(ROOT, 'data'))
    p.add_argument('--output_dir', default=os.path.join(ROOT, 'save_finetuned'))
    p.add_argument('--budget_hours', type=float, default=3.6,
                   help='wall-clock budget for everything this script does (benchmark + all runs)')
    p.add_argument('--epochs', type=int, default=0, help='fixed number of epochs (0 = fit the budget)')
    p.add_argument('--min_epochs', type=int, default=10)
    p.add_argument('--experiments', default='all', help='comma separated names, or "all"')
    p.add_argument('--compile', default='auto', choices=['auto', 'on', 'off'])
    p.add_argument('--bench_steps', type=int, default=100)
    p.add_argument('--eval_freq', type=int, default=5,
                   help='test-set top-1 every N epochs (+ always the last epoch); larger = less eval time')
    p.add_argument('--report_every', type=int, default=600, help='seconds between progress reports')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--list', action='store_true', help='list experiments and exit')
    a = p.parse_args()

    import torch
    n_gpus = torch.cuda.device_count()
    exps = experiments(n_gpus)
    if a.list:
        for e in exps:
            print(f'{e[0]:18s} {e[2]:6s} paper {e[3]:.2f}  {e[5]}')
        return
    if a.experiments != 'all':
        wanted = [s.strip() for s in a.experiments.split(',')]
        unknown = set(wanted) - {e[0] for e in exps}
        assert not unknown, f'unknown experiments: {unknown}'
        exps = [e for e in exps if e[0] in wanted]
    if n_gpus == 0:
        raise SystemExit('No CUDA GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.')
    if n_gpus < 2:
        exps = [e for e in exps if e[2] != 'ddp']
        log('only one GPU: DDP experiments are skipped')

    r = Runner(a)
    names = ', '.join(torch.cuda.get_device_name(i) for i in range(n_gpus))
    log(f'GPUs: {names}; budget {a.budget_hours}h; {len(exps)} experiments')

    sys.path.insert(0, ROOT)
    from utils.cifar_data import prepare_cifar
    prepare_cifar(a.datapath, 'CIFAR10')

    need_ddp = any(e[2] == 'ddp' for e in exps)
    compile_, single, ddp = r.benchmark(n_gpus, need_ddp)
    if need_ddp and ddp is None:
        log('DDP benchmark failed -> DDP experiments are skipped')
        exps = [e for e in exps if e[2] != 'ddp']
    epochs = a.epochs or r.plan_epochs(exps, n_gpus, single, ddp)
    est_single = r.est_run_sec('single', epochs, single, ddp)
    log(f'==> training every experiment for {epochs} epochs (paper: 100); '
        f'~{fmt_t(est_single)} per single-GPU run'
        + (f', ~{fmt_t(r.est_run_sec("ddp", epochs, single, ddp))} per DDP run' if ddp else ''))

    status = r.run_all(exps, n_gpus, epochs, compile_, single, ddp)
    info = (f'{epochs} epochs per run (paper: 100), {names}, AMP fp16, '
            f'{"torch.compile" if compile_ else "eager"}, GPU data pipeline. '
            f'Single-GPU throughput {single["img_per_sec"]:.0f} img/s'
            + (f', DDP {ddp["img_per_sec"]:.0f} img/s' if ddp else '')
            + f'. Total wall time {fmt_t(time.time() - r.t0)}.')
    r.report(exps, status, epochs, info)


if __name__ == '__main__':
    main()
