# Repository Guidelines

## Project Structure & Module Organization

This repository implements federated-learning experiments with backdoor attacks and defenses. Entry points at the repository root include `main_parallel.py` for parallel training, `traintest.py` for training/evaluation helpers, and `test.py` or `test_mirage.py` for focused experiments. Keep model definitions in `models/` and reusable data, attack, defense, logging, and configuration helpers in `utils/`. Experiment configurations belong in `conf/`; use descriptive YAML names such as `rb_cifar10_resnet20sm_static_dirichlet.yaml`. Batch launchers live in `scripts/`, while notebooks and `assets/` support analysis and datasets.

## Build, Test, and Development Commands

Create an isolated Python environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a configured experiment from the repository root:

```bash
python main_parallel.py --config conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --attack_type dba --debug
```

Use `bash scripts/get_base.sh -f scripts/<launcher>.sh` to start a batch launcher after reviewing its GPU IDs and arguments. Run `python test.py` or `python test_mirage.py` only when the selected script covers the code you changed.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase for constants. Keep CLI options and YAML keys aligned with the names consumed by `utils/options.py`. Prefer focused helpers in `utils/` over duplicating training or aggregation logic in an entry point. Preserve existing PyTorch tensor/device handling and make CPU/GPU placement explicit.

## Testing Guidelines

There is no unified test runner or coverage target. Validate changes with the narrowest relevant script and a short `--debug` run using a representative YAML configuration. For model, attack, or defense changes, record both clean-task accuracy and attack-success-rate effects; do not treat a successful process exit as experimental validation. Add test scripts using the `test_<feature>.py` pattern when practical.

## Commit & Pull Request Guidelines

The available history uses short imperative subjects such as `init`; continue with concise, imperative summaries, for example `Fix TSS device selection`. Keep commits scoped to one change. Pull requests should explain the experiment or bug addressed, list the exact configuration and command used, state hardware/runtime assumptions, and summarize accuracy/ASR results. Include plots or screenshots for visualization changes and link related issues when available.

## Configuration & Data Safety

Do not commit downloaded datasets, checkpoints, logs, credentials, or machine-specific paths. Treat YAML files as reproducibility artifacts: change one experimental factor at a time and name new configurations after the dataset, model, attack/defense, and data split.
