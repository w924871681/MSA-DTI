# MSA-DTI

Official code, frozen experimental datasets, protocols, finalized manuscript figure sources, and verified reproduction package for:

> **Response-Guided Model Selection and Structure-Preserving Adaptation for Few-Shot Digital Twin Instantiation**

MSA-DTI instantiates deployable predictive digital twins (DTs) for new computing centers from a shared reusable model bank under hard per-model operation and parameter limits. It couples **response-guided model selection (RGMS)** (linearized response estimation, dynamic-anchor historical bias correction, single-candidate reachable contender enumeration, complete-graph pairwise consensus, and direct-margin guarded switching) with **structure-preserving adaptation (SPA)** (5-state guarded early adaptation, 22-config support-only initial readout selection, optional in-place single-channel causal load routing, and multi-scale load-decomposed output calibration folded algebraically into the existing output layer with zero added parameters and zero added operations).

## Release & Version Notes

- **`v1.1`** (latest synchronized release): Includes the exact frozen RGMS (`720/720` live selection match) and SPA (`3/3` live Bitbrains E adaptation fixtures) implementations, three-tier automated verification (`code/reproduce.py verify`), the finalized manuscript figure sources (`paper_figures/`, reproducing main-article **Figs. 5–10** via `code/reproduce.py figures`), automated figure-input and render QA (`verify_inputs.py`, `verify_figures.py`), `Supplementary_Data_S1.csv`, and reference manuscript PDFs.

## Repository Structure

| Path | Description |
| --- | --- |
| `code/msa_dti/` | Exact standalone implementation of frozen RGMS (`selection.py`), SPA (`adaptation.py`), and frozen bank/budget config (`config.py`) |
| `code/native/src/msa_dti_native/` | Native PyTorch model architectures (MLP, TCN, GRU), causal feature extraction, prefix windowing, and source-training recipes |
| `code/protocol/` | Frozen cohort splits, task manifests (Bitbrains A–E, Azure V2, UFCG 2024–2025, Bitbrains Rnd6), and `frozen_config.json` |
| `code/data/` | Frozen experimental summary tables, 720-deployment & 216-external records, `rgms_dev_cache.npz` (720-deployment live RGMS verification cache), `spa_fixture/` (3 frozen Bitbrains E SPA verification instances), timing logs, and sensitivity audits |
| `paper_figures/` | Self-contained finalized manuscript figure package reproducing **Figs. 5–10** (`render_figures.py`), automated QA (`verify_inputs.py`, `verify_figures.py`), `Supplementary_Data_S1.csv`, licensed fonts (`Tinos` and `Liberation Serif`), and reference manuscript PDFs |
| `code/plotting/` | Table generator (`generate_paper_assets.py --tables-only`) reproducing main-article **Tables 1–5** and Supplementary tables (legacy v1.0/v1.1 figure builders and `table3_main_bitbrains.tex` are retained strictly as archived references superseded by Fig. 5 and `paper_figures/`) |
| `code/verification/` | Three-tier automated verification suite (`verify_results.py`) |
| `code/reproduce.py` | Unified CLI to verify frozen results + live RGMS/SPA parity, regenerate finalized Figs. 5–10 and LaTeX tables, and run Figure QA |
| `code/run_targets.py` | End-to-end RGMS + SPA target instantiation runner (supports `--verify-fixture` and full Bitbrains/Azure runs) |
| `code/train.py` | Bitbrains cohort preparation and native source-model training entry point |

## Quickstart: Verification, Finalized Figures (Figs. 5–10), and Tables

Keep the figure-rendering dependencies (`paper_figures/requirements.txt`) separate from the training/PyTorch environment (`code/requirements-train.txt`) so figure package updates do not alter the `v1.1` numerical verification environment.

### 1. Three-Tier Numerical & Algorithm Verification (`verify`)

```bash
python -m pip install -r code/requirements-figures.txt
# Install PyTorch (CPU or CUDA) from code/requirements-train.txt to include Tier 3 live SPA verification

# Tier 1: Frozen dataset integrity & key paper metrics (Bitbrains E, Azure V2, UFCG, Rnd6)
# Tier 2: Live RGMS selection across all 720 development deployments (720/720 exact match, max|margin diff| < 1e-12)
# Tier 3: Live SPA adaptation across 3 frozen Bitbrains E instances (orig, low1, legacy_special; +0 params, +0 MACs)
python code/reproduce.py verify
```

### 2. Regenerate Finalized Experimental Figures (Figs. 5–10) & LaTeX Tables

```bash
# In a dedicated figure environment matching paper_figures/requirements.txt:
python -m pip install -r paper_figures/requirements.txt

# Render the 6 finalized main experimental PDFs (Figs. 5-10) to work/paper/figures/
# and regenerate LaTeX tables to work/paper/tables/
python code/reproduce.py figures

# Run figure input check against code/data/ and compare rendered Figs. 5-10 against manuscript reference PDFs
python code/reproduce.py verify-figures

# Or run the standalone self-contained figure package directly from paper_figures/:
python paper_figures/verify_inputs.py .
python paper_figures/render_figures.py work/paper/figures_check
python paper_figures/verify_figures.py work/paper/figures_check
# Under the pinned Python 3.12 / Matplotlib 3.10.8 reference environment, add --strict-final-three
```

### Manuscript Figure & Table Map

| Asset | Source / Status | Notes |
| --- | --- | --- |
| **Fig. 5** (four error metrics on Bitbrains E) | `paper_figures/src/manuscript/plot_figures.py` (`main_comparison`) | Supersedes legacy Fig. 5 and archived `table3_main_bitbrains.tex` |
| **Fig. 6** (paired task MSE vs. PT-FT and Full-Val) | `paper_figures/src/bitbrains/plot_bitbrains.py` (`build_fig6`) | Reuses frozen Bitbrains task scores, deployment events, and Tinos fonts |
| **Fig. 7** (robustness across horizons, support sizes, budgets) | `paper_figures/src/bitbrains/plot_bitbrains.py` (`build_fig7_forest`) | Clustered bootstrap 95% CIs |
| **Fig. 8** (deployment feasibility and selection frequencies) | `paper_figures/src/bitbrains/plot_bitbrains.py` (`build_fig8_budget_feasibility`) | Frozen resource profiles and selection decisions |
| **Fig. 9** (adaptation effects by selector on Bitbrains E) | `paper_figures/src/manuscript/plot_figures.py` (`bitbrains_ablation`) | Supersedes legacy Fig. 9 |
| **Fig. 10** (paired risk reduction across 42 external units) | `paper_figures/src/manuscript/plot_adaptation_pairs.py` | 30 Azure subscriptions + 12 UFCG projects; supersedes legacy Figs. 10 & 11 |
| **Figs. 1–4** (system and method diagrams) | Reference final PDFs only (`paper_figures/reference/figures/fig1.pdf`–`fig4.pdf`) | Diagram artwork; matching drawing scripts not bundled |
| **Fig. S1** (candidate bank screening) | Reference final PDF only (`paper_figures/reference/figures/figS_bank.pdf`) | Final PDF artwork |
| **Fig. S2** (inference latency; `figS3.pdf` in TeX project) | Reference final PDF only (`paper_figures/reference/figures/figS3.pdf`) | Underlying 76 median/p95 pairs archived in `paper_figures/reference/supplement/latency.csv` |
| **Main Tables 1–5** & **Supplementary Tables S3–S11** | `code/plotting/generate_paper_assets.py --tables-only` | Main Tables 1–5 cover setup, comparison methods, onboarding cost (`table4_feasibility_cost.tex`), ablation (`table5_ablation.tex`), and external evaluation (`table6_external.tex`); `table3_main_bitbrains.tex` is archived legacy only. `paper_figures/Supplementary_Data_S1.csv` provides the 40 external cohort normalized risks. |

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

Code is released under the [MIT License](LICENSE). Derived Azure workload series remain subject to Microsoft's CC BY 4.0 data license. The bundled Tinos and Liberation Serif fonts follow the license notices in `code/plotting/fonts/`, `paper_figures/src/bitbrains/fonts/`, and `paper_figures/fonts/liberation/COPYRIGHT.txt`.
