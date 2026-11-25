# MSA-DTI · manuscript figure sources (v1.1)

This package regenerates the six experimental figures in the main article, Figures 5–10. It contains the plotting sources, their frozen inputs, licensed fonts, manuscript PDF references and Supplementary Data S1. The same files apply to the teacher-review copy with text and table-label edits.

## Rebuild

Use Python 3.11 or 3.12 in a fresh environment. The pinned package versions match the tested Python 3.12 environment. Run these commands from the extracted directory, using an empty output directory.

```sh
python -m pip install -r requirements.txt
python render_figures.py build/figures
python verify_figures.py build/figures
```

For the stricter comparison of Figures 5, 9 and 10, run:

```sh
python verify_figures.py build/figures --strict-final-three
```

Under the tested environment, these three files match the corresponding manuscript PDFs byte for byte. For Figures 6–8, the verifier compares page sizes, text and displayed values, element locations, vector paths, embedded rasters and page renders. PDF timestamps, library versions and font rendering can change PDF bytes across platforms. The main render command reads frozen inputs directly and requires neither PyTorch nor source-model training.

## File map

| Manuscript | Source included here | Input or evidence |
| --- | --- | --- |
| Fig. 5, four error metrics | `src/manuscript/plot_figures.py`, function `main_comparison` | `src/manuscript/medium_errors_x1000.csv` |
| Fig. 6, paired task errors | `src/bitbrains/plot_bitbrains.py`, function `build_fig6` | Frozen task scores and deployment events in `src/bitbrains/data/` |
| Fig. 7, robustness intervals | Same module, `build_fig7_forest` | Same two source files, clustered bootstrap |
| Fig. 8, limits and selection frequencies | Same module, `build_fig8_budget_feasibility` | Frozen architecture resource profiles and selection decisions |
| Fig. 9, adaptation by selector | `src/manuscript/plot_figures.py`, function `bitbrains_ablation` | `src/manuscript/normalized_risk.csv` |
| Fig. 10, paired changes by external unit | `src/manuscript/plot_adaptation_pairs.py`, called by `plot_figures.py` | `src/manuscript/adaptation_pairs.csv` |
| Figs. 1–4, system and method diagrams | Final PDFs in `reference/figures/fig1.pdf` through `fig4.pdf` | The supplied material establishes the final artwork. Matching original drawing scripts have not been verified. |
| Fig. S1, candidate bank | Final `reference/figures/figS_bank.pdf` | Final PDF artwork. A matching original drawing script has not been verified. |
| Fig. S2, inference latency | Final `reference/figures/figS3.pdf` | `reference/supplement/latency.csv` contains the 76 original median/p95 pairs. A matching original drawing script has not been verified. |

All 12 reference PDFs, all files in `src/manuscript/` and `Supplementary_Data_S1.csv` come directly from the authors’ v1.1 LaTeX archive (SHA-256 `35cc424b7cb34484bf1447b7bc27113674d638f01354d9c893851e5e42495263`). For the six new renders, compare against `reference/figures/`. Some internal filenames predate the current numbering.

The functions for Figures 6–8 come from `code/plotting/generate_paper_assets.py` in public commit `1438a1d4ca245f05eda1ef04f67b69b714deebbe`. The standalone module adapts file paths and the PDF export entry point. Its four inputs have the same file contents as the public `code/data/` copies. The original project and font licenses are in `src/bitbrains/` and `src/bitbrains/fonts/`. The Liberation Serif license is in `fonts/liberation/COPYRIGHT.txt`.

## Check against the public numerical records

```sh
python verify_inputs.py /path/to/MSA-DTI
```

Pass the local repository root containing `code/data/`. This checks three derived CSVs, the four frozen inputs for Figures 6–8 and the 40 normalized risks in Supplementary Data S1. The original extraction scripts in `src/manuscript/` also accept the experiment archive described in their README. Rebuilding from the adjacent plotting CSVs does not require that archive.

The source-data digest for the archived experiment bundle, which also contains the supplementary latency table, is `4044eb777260fa55967547a9d9a90a1c047d464c369878a11ca7aeb988e235e4`. Refer to `FILE_SHA256.txt` for the contents of this handoff. 
