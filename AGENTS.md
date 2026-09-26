# Repository Guidelines

## Project Structure & Module Organization

Personal notes plus two independent upstream Git clones:
- `code/minimind/`: text LLM; code in `model/`, `dataset/`, `trainer/`, `scripts/`; CLI is `eval_llm.py`.
- `code/minimind-v/`: VLM and primary showcase project; see `model/model_vlm.py`, `trainer/train_*_vlm.py`.
- `notes/`: MkDocs pages and study notes for advisor review.
- `results/`: per-run evidence for resume and demos.

Run Git inside the clone you change; `out/`, `.venv/`, `checkpoints/`, datasets, and weights are generated.

## Execution Environment

- Read code locally; run training and evaluation on AutoDL/Linux, recording commands and configuration.
- This WSL checkout is canonical; the `E:\notes` copy is a stale fallback.
- Use the web UI for learning questions; edit files only for concrete code or documentation tasks.
- After editing a clone under `code/`, run `gitnexus analyze --index-only <repo-path>`.

## Build, Test, and Development Commands

No compiled build step. Commands target AutoDL; run locally only when requested. Python 3.10+.

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

Multi-GPU: `torchrun --nproc_per_node N train_xxx.py` from `trainer/`.

## Coding Style & Naming Conventions

Use 4-space indentation and PEP 8 spacing: `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants. Order imports standard library, third-party, local. No formatter or linter is configured; avoid unrelated reformatting.

## Testing Guidelines

No automated suite or CI command is defined. For local syntax checks run `python -m compileall model dataset trainer scripts`; run real evaluation smoke tests on AutoDL. Add future tests under `tests/` as `test_<module>.py`.

## Results & Showcase Sync

Training output is resume evidence; never leave it only on AutoDL. Center showcase material on `minimind-v`.
- Copy back metrics (`loss`, `lr`, `ppl`), configs, logs, eval output, curves, and screenshots after each run.
- Store each run under `results/minimind-v/<run-name>/` following `results/minimind-v/README.md`: goal, GPU model and count, runtime, command, metrics.
- Push to `RXCCCCCC/minimind-v-research` in the same session; keep weights, checkpoints, and datasets out of Git.

## Commit & Pull Request Guidelines

Notes repo uses `docs:` or `chore:`; MiniMind clones use `[fix]`, `[update]` style subjects. Match the active repository's history.

PRs state the goal, affected modules, verification result, and linked issue. Include screenshots for UI or docs changes and note GPU or compatibility impact.
