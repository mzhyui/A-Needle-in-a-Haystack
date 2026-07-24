# A-Needle-in-a-Haystack (FL_Box)

Implementation of **[A Needle in a Haystack: Defending Federated Learning Backdoor Attacks via Orthogonal Subnetwork Pruning](https://ieeexplore.ieee.org/abstract/document/11571338/)**, published in **IEEE INFOCOM 2026**.

Federated-learning research environment for evaluating backdoor attacks and server-side defenses. The project builds on [LG-FedAvg](https://github.com/pliang279/LG-FedAvg) and combines attack, aggregation, detection, and model-analysis workflows.

## Included research components

- Attacks: DBA, DarkFed, 3DFed, invisible, edge, blend, and Mirage variants.
- Defenses: Krum, clipping, RLR, FLAME, tracer, TSS/GSS, and masked trigger inversion.
- Models and datasets: LeNet, VGG, ResNet-20, ViT; MNIST, Fashion-MNIST, CIFAR-10/100, and GTSRB.

External implementations integrated into this repository include [DBA](https://github.com/AI-secure/DBA), [FedSAM](https://github.com/debcaldarola/fedsam), [3DFed](https://github.com/haoyangliASTAPLE/3DFed), [DarkFed](https://github.com/hustweiwan/DarkFed), [FLAME](https://github.dev/zhmzm/FLAME), and [Neural Cleanse](https://github.com/bolunwang/backdoor).

## Layout

- `main_parallel.py`: primary federated-training entry point.
- `conf/`: reproducible YAML experiment configurations.
- `models/`: network and soft-mask implementations.
- `utils/`: data loading, local training, attack, defense, aggregation, and analysis helpers.
- `scripts/`: batch launchers for common experiment combinations.
- `assets/`: bundled trigger and auxiliary data.

## Setup

Install the pinned research dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

The supplied requirement pins are legacy PyTorch versions; use a compatible Python/CUDA environment before running GPU experiments.

## Run an experiment

Run from the repository root with a YAML configuration. Command-line flags override YAML values.

```bash
python main_parallel.py \
  --config conf/rb_cifar10_resnet20sm_static_dirichlet.yaml \
  --attack_type dba --debug --tss_threshold 0.7
```

For scripted batches, review GPU IDs and parameters first, then use:

```bash
bash scripts/get_base.sh -f scripts/batch_run_rb0_resnet_gtsrb.sh
```

## Experiment checks

Report clean-task accuracy, attack success rate, and local-model behavior together. Preserve the exact configuration, command, dataset split, random seed, and output directory with each result. The main open work is improving attack stability, TSS/GSS detection quality, GPU memory usage, and resume/test coverage.

## Citation

This repository contains the source code and experimental framework associated with the following publication. If you use this code, configurations, or results in your research, please cite:

```bibtex
@inproceedings{ma2026needle,
  author    = {Zihan Ma and Guangchi Liu and Xiangyu Xu and Shaofeng Li and Zhen Ling and Junzhou Luo},
  title     = {A Needle in a Haystack: Defending Federated Learning Backdoor Attacks via Orthogonal Subnetwork Pruning},
  booktitle = {IEEE INFOCOM 2026 - IEEE Conference on Computer Communications},
  year      = {2026},
  pages     = {1--10},
  doi       = {10.1109/INFOCOM59046.2026.11571338}
}
```
