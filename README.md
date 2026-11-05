# MSA-DTI

Code, data protocols and manuscript for the budget-constrained initialization of a digital twin at a new center. Bitbrains is the main experiment. Azure VM traces (2019 release) provide the external evaluation.

| Item | Where to start |
| --- | --- |
| Main paper and supplementary material | [Manuscript](manuscript.pdf) · [Supplement](supplement.pdf) · [Editable LaTeX project](latex.zip) |
| Experimental code | [Source training and development](code/train.py) · [Target adaptation, selection and scoring](code/run_targets.py) · [Native model implementations](code/native/src) |
| Frozen inputs and results | [Cohorts and task manifests](code/protocol) · [Paper data](code/data) · [Weights and per-window forecasts](https://github.com/w924871681/MSA-DTI/releases/tag/review-v1) |
| Report generation | [Figure and table scripts](code/plotting) · [Verification scripts](code/verification) |

## Reproduce the tables and figures

Start in the repository root. Use Python 3.11. For Linux or macOS the commands below create a virtual environment. On Windows use `.venv\Scripts\activate` after creating it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-figures.txt
python code/reproduce.py figures
```

This generates Figures 5–10 and S1–S3 plus 17 LaTeX table files and text fragments. The code checks the regenerated assets against the editable paper. Figures 1–4 and the two descriptive main-text tables are supplied in the LaTeX project. The assembled sources and figures are in `work/paper/`. Compile `main.tex` and `supplementary.tex` separately in Overleaf or with pdfLaTeX. Both final PDFs are also included at the top level.

## Independently check the published predictions

Download the **fastStorage** ZIP from the [Grid Workloads Archive](https://atlarge-research.com/gwa-t-12/) ([direct download](https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip)). Keep the original ZIP intact. The expected SHA-256 is `11313f528a0cbcbe57e63162f8ae5a41a9c7e7c1a79872e294ff3c5bbaa2e671`. Substitute its actual location in the following commands.

```bash
python code/reproduce.py scores --bitbrains /path/to/fastStorage.zip
python code/reproduce.py supplement --bitbrains /path/to/fastStorage.zip
python code/reproduce.py figures --from-verified
```

The first command downloads checksummed author forecasts, reconstructs the label windows and recalculates MSE, MAE and Worst-10% for eight methods, three budgets and 120 tasks in each dataset. It compares the selected architecture and numerical metrics task by task. The second command checks the supplemental selections, forecast errors, source checkpoints and timing records. It also verifies the preselection ablation and the additional Azure search runs. The last command feeds the recomputed scores back into the paper's reporting scripts. Resource profiles, historical timings and retained-bank calibration records have separate inputs. Generated outputs appear in `work/verified/` and `work/paper/`.

The primary PT-FT uses the fixed A39 backbone at every budget. Tight-budget predictions from the auxiliary budget-aware PT-FT run in the historical CSV are excluded from the primary comparison. The three NAS searches build their shortlists under each active budget. Azure Table 6 reports feasible outputs and uses the same 30 subscriptions at all three budgets.

## Rerun adaptation and selection

Use a separate **Python 3.11** environment with an NVIDIA GPU and driver supporting CUDA 12.1. The reported runs used a Tesla V100S, PyTorch 2.5.1+cu121 and the native GRU backend. Install the CUDA dependencies and load the frozen source weights.

```bash
python -m pip install -r code/requirements-train.txt
python code/reproduce.py fetch weights
python code/run_targets.py --dataset bitbrains --bitbrains /path/to/fastStorage.zip --tasks 1
python code/run_targets.py --dataset azure --tasks 1
```

Each smoke command fits **all eight methods at all three budgets** for one task. Remove `--tasks 1` to run all 120 tasks per dataset. Use `--method pt msa` to run a subset. Model choice uses the observed prefix. The runner saves the selected weights before opening future labels for scoring. It prints architecture mismatches and the maximum absolute error differences for each metric and writes complete task records in `work/reruns/`. The optional `--strict` switch applies the much tighter archived-forecast replay tolerance to a fresh run. GPU kernels and training libraries can change floating point results and occasionally change the selected architecture.

## Retrain from the public traces

On **Linux or WSL2 with CUDA**, start in an empty working copy with the training dependencies installed. The following path derives the source tensors, retrains source weights, screens architectures, selects the references, calibrates on D and evaluates E. It also retrains the baseline source components and selects the fixed backbones on C.

```bash
python code/train.py all --bitbrains /path/to/fastStorage.zip
python code/run_targets.py --trained work/training/export --dataset bitbrains --bitbrains /path/to/fastStorage.zip
python code/run_targets.py --trained work/training/export --dataset azure
```

The individual training stages are `prepare`, `msa`, `policy`, `search`, `pt`, `medet`, `scratch` and `hmeta`. The single-device runner uses one process per stage. The historical wall-clock numbers in Table 4 come from the original scheduling and recorded interruptions. Use the original measurements to verify that table and report new machine timings separately. [Training workers](code/experiments/reference) · [Verification scope](code/VALIDATION.md)

## Data access and attribution

For a new run of the supplemental source-pool, seed, optimizer, subset and hardware experiments, run `train.py prepare` as above, `python code/reproduce.py fetch weights`, then `python code/train.py supplement` on Linux/CUDA. The original control weights anchor the comparison. New panels and device timings are written to `work/training/supplement/panels/`.

- **Bitbrains.** Bitbrains IT Services Inc. provided the traces through the Grid Workloads Archive. This study uses GWA-T-12 fastStorage. See S. Shen, V. van Beek and A. Iosup, *Statistical Characterization of Business-Critical Workloads Hosted in Cloud Datacenters*, CCGrid 2015. [Provider and acknowledgement terms](https://atlarge-research.com/gwa-t-12/)
- **Azure.** The provider is Microsoft. Use the 2019 Azure VM trace (V2). The repository includes the derived CPU time series for the 30 selected subscriptions, with the other nine channels masked. These derived data are supplied under the provider's CC BY 4.0 data license. The extraction retains 3,000 consecutive five-minute readings per VM after the initial day. To check the extraction, download the 75 CPU shards named in the frozen manifest into one directory and run `python code/get_azure.py /path/to/shards`. The script verifies provider hashes, repeats the coverage screen on 280 prespecified candidates and compares all 120 task inputs. Please cite Microsoft's *Resource Central* paper when using the traces. [V2 documentation](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) · [Provider download list](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetLinksV2.txt) · [Data license](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE) · [Shard manifest](code/protocol/azure/shards.json)
- The supplied Tinos fonts include their license in [the plotting directory](code/plotting/fonts). Source, model and result files in this snapshot correspond to the manuscript PDFs at the repository root.

