# MSA-DTI

Official code, frozen experimental datasets, protocols, and verified reproduction package for:

> **Response-Guided Model Selection and Structure-Preserving Adaptation for Few-Shot Digital Twin Instantiation**

MSA-DTI instantiates deployable predictive digital twins (DTs) for new computing centers from a shared reusable model bank under hard per-model operation and parameter limits. It couples **response-guided model selection (RGMS)** (linearized response estimation, dynamic-anchor historical bias correction, single-candidate reachable contender enumeration, complete-graph pairwise consensus, and direct-margin guarded switching) with **structure-preserving adaptation (SPA)** (5-state guarded early adaptation, 22-config support-only initial readout selection, optional in-place single-channel causal load routing, and multi-scale load-decomposed output calibration folded algebraically into the existing output layer with zero added parameters and zero added operations).

## Repository Structure

| Path | Description |
| --- | --- |
| `code/msa_dti/` | Exact standalone implementation of frozen RGMS (`selection.py`), SPA (`adaptation.py`), and frozen bank/budget config (`config.py`) |
| `code/native/src/msa_dti_native/` | Native PyTorch model architectures (MLP, TCN, GRU), causal feature extraction, prefix windowing, and source-training recipes |
| `code/protocol/` | Frozen cohort splits, task manifests (Bitbrains A–E, Azure V2, UFCG 2024–2025, Bitbrains Rnd6), and `frozen_config.json` |
| `code/data/` | Frozen experimental summary tables, 720-deployment & 216-external records, `rgms_dev_cache.npz` (720-deployment live RGMS verification cache), `spa_fixture/` (3 frozen Bitbrains E SPA verification instances), timing logs, and sensitivity audits |
| `code/plotting/` | Scripts (`generate_paper_assets.py`) and Tinos fonts reproducing Figures 5–11 and Tables 1–6 & S3–S11 |
| `code/verification/` | Three-tier automated verification suite (`verify_results.py`) |
| `code/reproduce.py` | Unified CLI to verify frozen results + live RGMS/SPA parity and regenerate all experimental figures and tables |
| `code/run_targets.py` | End-to-end RGMS + SPA target instantiation runner (supports `--verify-fixture` and full Bitbrains/Azure runs) |
| `code/train.py` | Bitbrains cohort preparation and native source-model training entry point |

## Quickstart: Three-Tier Verification & Paper Asset Reproduction

Requires Python 3.10+ (~15 seconds total):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r code/requirements-figures.txt
# Install PyTorch (CPU or CUDA) from code/requirements-train.txt to include Tier 3 live SPA verification

# Run three-tier verification:
#   Tier 1: Frozen dataset integrity & key paper metrics (Bitbrains E, Azure V2, UFCG, Rnd6)
#   Tier 2: Live RGMS selection across all 720 development deployments (720/720 exact match, max|margin diff| < 1e-12)
#   Tier 3: Live SPA adaptation across 3 frozen Bitbrains E instances (orig, low1, legacy_special; +0 params, +0 MACs)
python code/reproduce.py verify

# Regenerate all experimental figures (Figs. 5-11) and LaTeX tables (Tables 1-6, S3-S11)
python code/reproduce.py figures

# Or run both in one step
python code/reproduce.py all
```

Generated figures and LaTeX tables are written to `work/paper/figures/` and `work/paper/tables/`.

## Core Algorithm Modules (`code/msa_dti/`)

- **Response-Guided Model Selection** (`code/msa_dti/selection.py`):
  - `center_scale` & `history_reference_scale`: cross-center scale $s_c$ and budget reference scale $C_b$
  - `anchor_pair_features` & `calibrated_scores`: dynamic target anchor $q_{\mathrm{anc}}$, 35-D anchor-relative paired features, and historical-neighbor bias correction (Stage A)
  - `reachable_set`: single-candidate historical subset replacement enumeration $\mathcal{R}_c$
  - `pairwise_direct_margins`, `consensus_scores`, `guarded_replacement`, `select_candidate`: three-group standardized L1 pairwise margin estimation, complete-graph consensus ranking on $\mathcal{R}_c$, and direct-margin switching guard (Stage B)
- **Structure-Preserving Adaptation** (`code/msa_dti/adaptation.py`):
  - `early_stop_base02`: 5-state guarded early adaptation (`b50`, `b25`, `b10`, midpoint `bW`, `a0`)
  - `fit_initial_readout_r62` & `r63_native_proposals`: 22-config support-only initial readout selection and 161-candidate single-scale/joint readout bank for unmodified backbones
  - `channel_importance`, `routable_channels`, `rewrite_single_channel`, `topology_signature`: leave-one-channel-out ridge importance scoring and exact in-place single-channel causal load routing across MLP, TCN, and GRU layers with SHA-256 topology verification (+0 parameters, +0 operations)
  - `reconstruct_response`, `multi_scale_calibration`, `adapt_model_spa`: multi-scale load-decomposed calibration (`orig`: 1,089 candidates; `low1`/`low5`: 928 candidates) folded algebraically into the existing linear output layer

## End-to-End Source Training & Target Instantiation (GPU)

Download the original **GWA-T-12 fastStorage** archive from the [Grid Workloads Archive](https://atlarge-research.com/gwa-t-12/) (SHA-256: `11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671`):

```bash
python -m pip install -r code/requirements-train.txt

# Verify bundled 3-instance SPA fixture directly
python code/run_targets.py --verify-fixture

# Prepare verified Bitbrains cohorts and run native source training
python code/train.py prepare --bitbrains /path/to/fastStorage.zip
python code/train.py c1 --bitbrains /path/to/fastStorage.zip

# Run frozen RGMS + SPA target instantiation using source checkpoints
python code/run_targets.py --dataset bitbrains --bitbrains /path/to/fastStorage.zip --weights-dir /path/to/source_weights --output-dir work/runs/bitbrains
python code/run_targets.py --dataset azure --weights-dir /path/to/source_weights --output-dir work/runs/azure
```

## Public Datasets & Attribution

- **Bitbrains GWA-T-12 fastStorage** (primary in-domain evaluation & held-out Rnd6 queue): Grid Workloads Archive, Shen et al., *CCGrid 2015*.
- **Microsoft Azure Public Dataset V2** (cross-platform external evaluation): 30 subscriptions with 9 unavailable raw telemetry channels masked; derived CPU series included under CC BY 4.0.
- **UFCG OpenStack 2024–2025** (prospective private-cloud evaluation): 12 qualifying projects sealed prior to unlocking evaluation windows (`code/protocol/ufcg_cases.csv`).

## License

Code is released under the [MIT License](LICENSE). Derived Azure workload series remain subject to Microsoft's CC BY 4.0 data license. The bundled Tinos fonts follow the Apache 2.0 / OFL notices in `code/plotting/fonts/`.
