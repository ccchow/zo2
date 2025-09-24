# Repository Guidelines
Use this reference when contributing to ZO2.

## Project Structure & Module Organization
- `zo2/` packages configs, models, trainers, and utilities; ship core logic here.
- `example/mezo_runner/` provides canonical jobs; mirror its layouts when adding tasks or optimizers.
- `script/` covers orchestration helpers, `test/` holds smoke and benchmark suites (`installation`, `mezo_sgd`), while `tutorial/` and `analysis/` store notebooks tied to public APIs.

## Build, Test, and Development Commands
- `conda env create -f env.yml && conda activate zo2` sets up the supported CUDA/PyTorch stack.
- `pip install -e .` keeps the package editable; rerun after dependency updates.
- `bash test/installation/run_all_tests.sh` triggers import, CUDA, and minimal-training smoke tests, logging to `test_results_*`.
- `bash test/mezo_sgd/hf_opt/test_{memory,speed,acc}_train.sh` measures GPU footprint, throughput, and accuracy for OPT baselines.
- For end-to-end validation, export the variables referenced in `example/mezo_runner/mezo.sh` (e.g. `MODEL=facebook/opt-2.7b`) and run `CUDA_VISIBLE_DEVICES=0 bash example/mezo_runner/mezo.sh`.

## Coding Style & Naming Conventions
- Target Python ≥3.11, four-space indentation, and PEP 8 formatting; wrap lines near 100 characters.
- Modules and functions use snake_case; classes adopt CapWords (`ZOConfig`, `BaseZOModel`).
- Document new public APIs with short docstrings and type hints, especially around offloading behavior.
- Reuse helpers in `zo2/utils`, group imports standard/third-party/local, and avoid ad-hoc logging in library code.

## Testing Guidelines
- Place environment checks in `test/installation`; put algorithm-specific regressions under `test/mezo_sgd` or a new descriptive subfolder.
- Name Python tests `test_<topic>.py` and shell flows `test_<metric>_*.sh`; ensure they run from the repo root.
- Note GPU (≈18GB for OPT-175B) and host RAM assumptions at the top of long-running tests, and attach log directories when sharing results.

## Commit & Pull Request Guidelines
- Follow the existing history style: short, present-tense summaries (`update main README`, `fix qwen3 eval bugs`) with optional scope prefixes.
- Keep commits focused and avoid mixing experiments, API revisions, and documentation tweaks.
- PRs should include a concise narrative, reproduction commands, hardware context, linked issues, and artefacts from the scripts above.

## Environment & Offloading Notes
- Keep config defaults aligned with `env.yml`; flag new CUDA or driver needs before merging.
- Exclude credentials and dataset paths; document required environment variables in `example/` READMEs.
- Flag hardware deviations early—large-model runs assume ~18GB GPU VRAM plus ample CPU memory for offloading.
