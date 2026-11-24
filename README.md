# MSA-DTI

Official code, frozen experimental datasets, protocols, and reproduction package for:

> **Response-Guided Model Selection and Structure-Preserving Adaptation for Few-Shot Digital Twin Instantiation**

MSA-DTI instantiates deployable predictive digital twins (DTs) for new computing centers from a shared reusable model bank under hard per-model operation and parameter limits. It couples **response-guided model selection** (linearized response estimation, historical bias correction, pairwise consensus, and direct-margin guarded switching) with **structure-preserving adaptation** (guarded early adaptation, in-place single-channel causal load routing, and multi-scale output calibration with zero added parameters and zero added operations).

## Repository Structure

| Path | Description |
| --- | --- |
| `code/msa_dti/` | Core standalone implementation of MSA-DTI (`config.py`, `selection.py`, `adaptation.py`) |
| `code/native/` | Native PyTorch model architectures (MLP, TCN, GRU), source-training recipes, and protocols |
| `code/protocol/` | Frozen cohort splits, task manifests (Bitbrains, Azure V2, UFCG 2024–2025, Bitbrains Rnd6), and `frozen_config.json` |
| `code/data/` | Frozen experimental summary tables, 720-deployment records, timing logs, and sensitivity audits |
| `code/plotting/` | Scripts (`generate_paper_assets.py`) and Tinos fonts reproducing Figures 5–11 and Tables 1–6 & S3–S11 |
| `code/verification/` | Integrity and numerical verification suite (`verify_results.py`) |
| `code/reproduce.py` | Unified CLI to verify frozen results and regenerate all experimental figures and tables |

Compiled manuscript PDFs (`main.pdf`, `supplementary.pdf`), the editable LaTeX project (`MSA-DTI-latex-project.zip`), finalized vector figures (`MSA-DTI-figures-final.zip`), and the standalone frozen data/script archive (`MSA-DTI-experiment-data-and-scripts.zip`) are attached to [Release v1.0](https://github.com/w924871681/MSA-DTI/releases/tag/v1.0).

## Quickstart: Verify Results & Regenerate Paper Figures (CPU)

Requires Python 3.10+ (CPU only, ~10 seconds):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r code/requirements-figures.txt

# Verify frozen experimental artifacts and core modules
python code/reproduce.py verify

# Regenerate all experimental figures (Figs. 5-11) and LaTeX tables (Tables 1-6, S3-S11)
python code/reproduce.py figures
```

Generated figures and LaTeX tables are written to `work/paper/figures/` and `work/paper/tables/`.

## Core Algorithm Modules

- **Response-Guided Model Selection** (`code/msa_dti/selection.py`):
  - `center_scale`: cross-center scale normalization
  - `calibrated_scores`: historical-neighbor bias correction and initial candidate selection
  - `pairwise_direct_margins`: standardized three-group distance matching and direct pairwise margin estimation
  - `consensus_scores` & `guarded_replacement`: complete-graph consensus ranking with direct-margin switching guard
- **Structure-Preserving Adaptation** (`code/msa_dti/adaptation.py`):
  - `channel_importance`: leave-one-channel-out ridge reconstruction importance scoring
  - `rewrite_single_channel` & `topology_signature`: exact in-place single-channel routing across MLP, TCN, and GRU layers with SHA-256 topology verification (+0 parameters, +0 operations)
  - `reconstruct_response` & `multi_scale_calibration`: unlabeled response reconstruction and multi-scale load-decomposed readout folded algebraically into the existing linear output layer

## Full Source Training & Target Adaptation (GPU)

For end-to-end training from public cloud traces, install PyTorch with CUDA (`code/requirements-train.txt`) and download the original **GWA-T-12 fastStorage** archive from the [Grid Workloads Archive](https://atlarge-research.com/gwa-t-12/) (SHA-256: `11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671`):

```bash
python -m pip install -r code/requirements-train.txt
python code/train.py prepare --bitbrains /path/to/fastStorage.zip
python code/train.py all --bitbrains /path/to/fastStorage.zip
python code/run_targets.py --trained work/training/export --dataset bitbrains --bitbrains /path/to/fastStorage.zip --output-dir work/runs/full
python code/run_targets.py --trained work/training/export --dataset azure --output-dir work/runs/full
```

## Public Datasets & Attribution

- **Bitbrains GWA-T-12 fastStorage** (primary in-domain evaluation & held-out Rnd6 queue): Grid Workloads Archive, Shen et al., *CCGrid 2015*.
- **Microsoft Azure Public Dataset V2** (cross-platform external evaluation): 30 subscriptions with 9 unavailable raw telemetry channels masked; derived CPU series included under CC BY 4.0.
- **UFCG OpenStack 2024–2025** (zero-tuning prospective private-cloud evaluation): 12 qualifying projects sealed prior to unlocking evaluation windows (`code/protocol/ufcg_cases.csv`).

## License

Code is released under the [MIT License](LICENSE). Derived Azure workload series remain subject to Microsoft's CC BY 4.0 data license. The bundled Tinos fonts follow the Apache 2.0 / OFL notices in `code/plotting/fonts/`.
