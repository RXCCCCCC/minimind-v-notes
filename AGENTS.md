# Repository Guidelines

## Project Structure & Module Organization

This workspace combines a notes repository with two independent Git clones:
- `code/minimind/`: text LLM; core code is under `model/`, `dataset/`, `trainer/`, and `scripts/`, with `eval_llm.py` as the CLI.
- `code/minimind-v/`: vision-language model; see `model/model_vlm.py`, `trainer/train_*_vlm.py`, and `eval_vlm.py`.
- `notes/minimind-2/`: MkDocs pages under `docs/` and assets under `docs/images/`.

Run Git commands inside the clone being changed. Treat `out/`, `.venv/`, `checkpoints/`, datasets, and model weights as generated files.

## Execution Environment

- Use this workspace mainly for reading and reviewing code locally; do not assume training, evaluation, or WebUI runs locally.
- Run training and heavy evaluation on AutoDL/Linux and record the required command and configuration.
- Use the web UI for learning questions; change local files only for concrete code or documentation tasks.
- After changing `code/minimind/` or `code/minimind-v/`, run `gitnexus analyze --index-only <repo-path>` before finishing.

## Build, Test, and Development Commands

No compiled build step exists. The commands below target the remote workflow; run them locally only when requested. Use Python 3.10+.

```powershell
cd code/minimind
pip install -r requirements.txt
cd trainer
python train_pretrain.py
python train_full_sft.py
cd ..
python eval_llm.py --weight full_sft
```

```powershell
cd code/minimind-v
pip install -r requirements.txt
cd trainer
python train_sft_vlm.py --epochs 2 --from_weight llm
cd ..
python eval_vlm.py --weight sft_vlm
```

For multi-GPU training, run `torchrun --nproc_per_node N train_xxx.py` from the relevant `trainer/` directory.

## Coding Style & Naming Conventions

Use 4-space Python indentation and PEP 8 spacing. Name functions and variables `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`; examples include `MiniMindConfig` and `VLMDataset`. Group imports as standard library, third-party, then local modules. No formatter or linter is configured; avoid unrelated formatting churn.

## Testing Guidelines

No automated test suite, coverage threshold, or CI command is defined. For lightweight local checks, run `python -m compileall model dataset trainer scripts eval_llm.py` (or VLM equivalents). Run real evaluation or training smoke tests on AutoDL. Add future tests under `tests/` as `test_<module>.py`.

## Commit & Pull Request Guidelines

The notes repository uses `docs: ...` and `chore: ...`; MiniMind clones commonly use `[fix]`, `[update]`, `[feat]`, `[perf]`, and `[refactor]`. Match the active repository's history.

PRs should state the goal, affected modules, verification command and result, and linked issue. Include screenshots for WebUI or documentation changes; note GPU, checkpoint, or compatibility impact. Do not commit credentials, datasets, checkpoints, or new weights unless the upstream repository requires them.
