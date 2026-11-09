# A Poor Vision Transformer with Competitive Performance on Small-scale Datasets by Low-Rank Attensions and Low-Dimensional Patch Tokens

Our work real the redundancy among ViTs and prove that low-rank attention and low-dimensional patch tokens can achieve competitive performance on small-scale datasets.

> **Abstract:** 
> Vision Transformers (ViTs) have demonstrated remarkable success on large-scale datasets, but their performance on smaller datasets often falls short of convolutional neural networks (CNNs). This paper explores the design and optimization of Tiny ViTs for small datasets, using CIFAR-10 as a benchmark. We systematically evaluate the impact of data augmentation, patch token initialization, low-rank compression, and multi-class token strategies on model performance. Our experiments reveal that low-rank compression of queries in Multi-Head Latent Attention (MLA) incurs minimal performance loss, indicating redundancy in ViTs. Additionally, introducing multiple CLS tokens improves global representation capacity, boosting accuracy. These findings provide a comprehensive framework for optimizing Tiny ViTs, offering practical insights for efficient and effective designs. Code is available at \url{https://github.com/erow/PoorViTs}.



# Reproducing the paper on Kaggle (2x T4, < 4 h)

1. Upload `notebooks/poorvits_kaggle.ipynb` to Kaggle, set *Accelerator: GPU T4 x2* and *Internet: On*.
2. *Save Version -> Save & Run All*. The notebook clones this repo, downloads CIFAR-10 from the Hugging Face CDN,
   trains all 16 experiments of Tables 2 & 3 and reports their CIFAR-10 test accuracy.

Locally / any machine:

```bash
pip install -r requirements.txt
python run_experiments.py --list                      # experiments + paper accuracy
python run_experiments.py --budget_hours 3.6          # epochs are sized to fit the budget
python run_experiments.py --epochs 100 --experiments baseline   # exact paper schedule for one run
python evaluate.py --output_dir save_finetuned        # optional: re-test saved checkpoints
```

What changed compared to the original code (so that it runs fast and correctly on 2x T4):

* **GPU data pipeline** (`utils/gpu_aug.py`, default `--data_pipeline gpu`): CIFAR is kept on the GPU and flip/crop,
  the CIFAR10 AutoAugment policy, normalisation and random erasing are applied batch-wise with the same
  probabilities/magnitudes as the PIL version (Kaggle has only 4 CPU cores). `--data_pipeline cpu` keeps the original one.
* fast dataset download (`utils/cifar_data.py`), AMP + optional `torch.compile`, no per-step GPU syncs.
* fixes: wandb is optional (`--wandb`), crash at the end of single-GPU runs (`args.rank`, `acc1`), mixup ran without
  autocast, DDP uses NCCL instead of gloo, whitening init is applied before DDP wrapping, the last epoch is always
  validated, sin-cos position embedding with several CLS tokens, ViT depth = 9 as in the paper, lr = 0.002 as in the paper.
* `run_experiments.py` (scheduler + summary table/plots) and `evaluate.py` (optional re-test of checkpoints).
* only the metrics the paper reports are computed: test top-1 every `--eval_freq` (5) epochs + the last epoch,
  runtime, and the mean train loss (GPU-side, one sync per epoch). Train accuracy, validation loss, the
  separate re-evaluation pass and per-step host syncs were removed.

Assumptions where the paper is not explicit: MLA latent dim d_c = 64; "bsN" in the DDP rows is the global batch size;
Lion uses lr 2e-4 / weight decay 0.5 (AdamW lr/10, wd x10). With fewer than 100 epochs absolute accuracies are
lower than in the paper; compare the differences to the baseline.

# main results

|         | Baseline | Augmentation |        |         | Patch |       |  MLA |       |       |       |      |  DDP  |        |              |
|---------|:--------:|:------------:|:------:|:-------:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:------:|:------------:|
|         |          |      -aa     | -mixup | -cutmix |  sin  | Witen |  qkv |   kv  |   qk  |   q   |   k  | bs256 | bs1024 | Lion(bs1024) |
| Runtime |   4658   |     4684     |  4656  |   4651  |  4687 |  4676 | 4805 |  4756 |  4785 |  4716 | 4711 |  3582 |  1371  |     1545     |
| Val/Acc |   93.65  |     92.14    |  93.68 |  92.77  | 93.43 | 93.28 | 91.8 | 92.43 | 92.42 | 93.32 | 92.5 | 93.44 |  92.09 |     92.38    |


## Citation
If you use our work, please consider citing:
```bibtex 

```
<hr>


## References
Our code is build on the repositories of [Train Vision Transformer on Small-scale Datasets](https://github.com/hananshafi/vits-for-small-scale-datasets) and [Vision Transformer for Small-Size Datasets](https://github.com/aanna0701/SPT_LSA_ViT). We thank them for releasing their code.

<hr>

  
