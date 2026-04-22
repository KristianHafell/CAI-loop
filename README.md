# CAI Critique–Revision Loop

**Overview**

This repository implements a single-model Critique–Revision (CAI) loop where one LM generates, critiques, revises, and self-evaluates responses. It produces a JSON `results.json` file suitable for visualization.

**Prerequisites**

- **Python:** 3.8 or newer
- **Ollama:** running locally (`ollama serve`) with models `llama3.1:8b` and `llama-guard3:8b` available
- **HuggingFace token:** stored in a `.env` file as `HF_TOKEN` (optional for dataset access)

**Quick Setup**

1. Create a `.env` with `HF_TOKEN=your_token` (if using HF dataset).
2. Source the setup script to install deps and verify services:

```
source setup.sh
```

**Usage**

- Run an experiment (example):

```
python cai_loop.py --n 3 --loops 1
```

- Plot results from one or more runs:

```
python plot_results.py --files results.json
```

**Files**

- **cai_loop.py**: core loop implementation; extracts prompts, runs generation → critique → revision cycles, and self-scores. See [cai_loop.py](cai_loop.py).
- **plot_results.py**: plotting utilities to visualize HH score progression across revisions. See [plot_results.py](plot_results.py).
- **setup.sh**: environment helper that checks Python, installs `datasets` and `requests`, verifies Ollama, and pulls models. See [setup.sh](setup.sh).
- **results.json**: example output produced by runs. See [results.json](results.json).

**Notes & Tips**

- The loop uses Ollama HTTP API at `http://localhost:11434`—ensure Ollama is running and models are pulled before running experiments.
- If you lack HF access, `--hf-token` can be omitted; provide custom prompts instead.
- Adjust `LM_MODEL` and `OLLAMA_BASE` in `cai_loop.py` to target different local endpoints or models.
