# MSA-DTI

Code, data protocols and manuscript for the budget-constrained initialization of a digital twin at a new center. Bitbrains is the main experiment. Azure VM traces (2019 release) provide the external evaluation.

| Item | Where to start |
| --- | --- |
| Main paper and supplementary material | [Manuscript](manuscript.pdf) · [Supplement](supplement.pdf) · [Editable LaTeX project](latex.zip) |
| Experimental code | [Source training and development](code/train.py) · [Target adaptation, selection and scoring](code/run_targets.py) · [Native model implementations](code/native/src) |
| Frozen inputs and results | [Cohorts and task manifests](code/protocol) · [Paper data](code/data) · [Weights and per-window forecasts](https://github.com/w924871681/MSA-DTI/releases/tag/review-v1) |
| Report generation | [Figure and table scripts](code/plotting) · [Verification scripts](code/verification) |

The [review-v1 release](https://github.com/w924871681/MSA-DTI/releases/tag/review-v1) contains the source and checksummed experimental assets. Run the commands below from the extracted repository root. Download the original **fastStorage** ZIP from the [Grid Workloads Archive](https://atlarge-research.com/gwa-t-12/) ([direct download](https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip)). Replace `/path/to/fastStorage.zip` with its location. Its SHA-256 is `11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671`. The selected Azure inputs are included.

## 1. Check the published results and reproduce the paper figures

Use Python 3.11. The following commands need only a CPU. On Windows activate the environment with `.venv/Scripts/activate`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-figures.txt
python code/reproduce.py scores --bitbrains /path/to/fastStorage.zip
python code/reproduce.py supplement --bitbrains /path/to/fastStorage.zip
python code/reproduce.py figures --from-verified
```

The first two commands check the downloaded assets (or use the copies in `work/assets/`), reconstruct future labels and verify results. The third regenerates and checks nine experimental figures (5–10 and S1–S3) and 17 LaTeX tables or text fragments against the paper. The four illustrations and two descriptive tables are supplied in the editable LaTeX project. For figures directly from the supplied summaries, run `python code/reproduce.py figures`. Output is in `work/verified/` and `work/paper/`. Compile `main.tex` and `supplementary.tex` separately there.

The main PT-FT comparison uses the fixed A39 backbone at each budget. The three NAS searches form their shortlists under the active budget. The Azure evaluation uses the same 30 subscriptions at all three budgets.

## 2. Train on the public traces and report your own results

Use a separate Python 3.11 environment on Linux with an NVIDIA GPU. The original experiment used a Tesla V100S, PyTorch 2.5.1 with CUDA 12.1 and the native GRU backend. From a clean working copy, train the source models, select the references on the historical cohorts, calibrate the switching margin and evaluate the two target datasets. Use a separate output directory for each experiment.

```bash
python -m pip install -r code/requirements-train.txt -r code/requirements-figures.txt
python code/train.py all --bitbrains /path/to/fastStorage.zip
python code/run_targets.py --trained work/training/export --dataset bitbrains --bitbrains /path/to/fastStorage.zip --output-dir work/runs/full
python code/run_targets.py --trained work/training/export --dataset azure --output-dir work/runs/full
python code/reproduce.py figures --from-run work/runs/full
```

The full report checks for eight methods, three budgets and the same 120 tasks per method in each dataset. It writes three files under `work/runs/full/report/`:

- `task_scores.csv` lists each selected architecture, its scores, parameter count, operation count and whether it met both limits.
- `performance_and_feasibility.csv` gives the task counts, feasibility counts and errors by method and budget. Both all-output and feasible-only errors are included. A blank feasible-only metric means that no selected model met both limits.
- `performance_and_feasibility.pdf` plots prediction error and deployment coverage from these results. Published timings, ablations and calibration records have separate inputs.

### Quick GPU check with the published source weights

To test target adaptation, selection, scoring and plotting on one task per dataset, use the same training and plotting dependencies and run:

```bash
python code/reproduce.py fetch weights
python code/run_targets.py --dataset bitbrains --bitbrains /path/to/fastStorage.zip --tasks 1 --output-dir work/runs/smoke
python code/run_targets.py --dataset azure --tasks 1 --output-dir work/runs/smoke
python code/reproduce.py figures --from-run work/runs/smoke --partial
```

This example fits all eight methods under all three budgets. The report labels the number of tasks actually evaluated. Remove `--tasks 1` to evaluate all 120 tasks per dataset and omit `--partial` for a full report. The runner also accepts `--method pt msa` for a smaller comparison. Use the same task and method coverage across budgets and keep distinct runs in separate directories. GPU and library differences may affect fitted errors or the selected architecture; `--strict` checks them against the archived predictions at replay tolerance.

For a source-training check, run `python code/train.py prepare --bitbrains /path/to/fastStorage.zip` and `python code/train.py source-smoke` to fit one architecture for 50 epochs. For the main ablation, run `python code/run_ablation.py --bitbrains /path/to/fastStorage.zip` after fetching weights. Its four arms share the medium-budget tasks and the A57 reference; `--tasks 1` selects one task.

The individual training stages are `prepare`, `msa`, `policy`, `search`, `pt`, `medet`, `scratch` and `hmeta`. The single-device runner uses one process per stage. The historical wall-clock numbers in Table 4 come from the original scheduling and recorded interruptions. Use the original measurements to verify that table and report new machine timings separately. [Training workers](code/experiments/reference) · [Verification scope](code/VALIDATION.md)

For a new run of the supplemental source-pool, seed, optimizer, subset and hardware experiments, run `train.py prepare` as above, `python code/reproduce.py fetch weights`, then `python code/train.py supplement` on Linux/CUDA. The original control weights anchor the comparison. New panels and device timings are written to `work/training/supplement/panels/`.

## Data access and attribution

- **Bitbrains.** Bitbrains IT Services Inc. provided the traces through the Grid Workloads Archive. This study uses GWA-T-12 fastStorage. See S. Shen, V. van Beek and A. Iosup, *Statistical Characterization of Business-Critical Workloads Hosted in Cloud Datacenters*, CCGrid 2015. [Provider and acknowledgement terms](https://atlarge-research.com/gwa-t-12/)
- **Azure.** The provider is Microsoft. Use the 2019 Azure VM trace (V2). The repository includes the derived CPU time series for the 30 selected subscriptions, with the other nine channels masked. These derived data are supplied under the provider's CC BY 4.0 data license. The extraction retains 3,000 consecutive five-minute readings per VM after the initial day. To check the extraction, download the 75 CPU shards named in the frozen manifest into one directory and run `python code/get_azure.py /path/to/shards`. The script verifies provider hashes, repeats the coverage screen on 280 prespecified candidates and compares all 120 task inputs. Please cite Microsoft's *Resource Central* paper when using the traces. [V2 documentation](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) · [Provider download list](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetLinksV2.txt) · [Data license](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE) · [Shard manifest](code/protocol/azure/shards.json)
- The supplied Tinos fonts include their license in [the plotting directory](code/plotting/fonts). Source, model and result files in this snapshot correspond to the manuscript PDFs at the repository root.

